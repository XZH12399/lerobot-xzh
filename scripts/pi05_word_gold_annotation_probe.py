#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in [SCRIPT_ROOT, SRC_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_word.modeling_pi05_word import PI05WordPolicy
from pi05_word_single_step_probe import (  # noqa: E402
    SampleRecord,
    SingleStepLabel,
    build_config,
    build_state_normalizer,
    extract_hidden_features,
    resolve_dataset_indices,
    resolve_local_hf_snapshot,
    set_seed,
    summarize_selection,
    train_probe,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


def log_progress(message: str) -> None:
    print(f"[pi05_word_gold_probe] {message}", file=sys.stderr, flush=True)


def load_gold_records(annotation_manifest: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for line in annotation_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        annotation = row["annotation"]
        records.append(
            SampleRecord(
                dataset_index=-1,
                absolute_index=int(row["absolute_index"]),
                episode_index=int(row["episode_id"]),
                task_index=int(row["task_index"]),
                task=row["instruction"],
                label=SingleStepLabel(
                    primitive=str(annotation["primitive"]),
                    target=str(annotation["target"]),
                    relation=str(annotation["relation"]),
                ),
            )
        )
    return records


def split_indices_by_target(records: list[SampleRecord], val_ratio: float, seed: int) -> tuple[list[int], list[int], dict]:
    rng = random.Random(seed)
    target_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        target_to_indices[record.label.target].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    split_summary = {"singleton_targets_kept_in_train": 0, "targets_with_val": 0}

    for target, indices in sorted(target_to_indices.items()):
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            split_summary["singleton_targets_kept_in_train"] += 1
            continue
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
        split_summary["targets_with_val"] += 1

    return sorted(train_indices), sorted(val_indices), split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe pi05_word hidden states using cleaned gold annotations.")
    parser.add_argument(
        "--annotation-manifest",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence.jsonl"),
    )
    parser.add_argument("--dataset-root", default="/home/ct_24210860031/data/libero")
    parser.add_argument("--dataset-repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--checkpoint", default="lerobot/pi05_libero_base")
    parser.add_argument("--tokenizer", default="google/paligemma-3b-pt-224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--feature-batch-size", type=int, default=6)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/pi05_word_validation/pi05_word_gold_probe_high_confidence.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    annotation_manifest = args.annotation_manifest.resolve()
    records = load_gold_records(annotation_manifest)
    if len(records) < 8:
        raise RuntimeError(f"Not enough gold records found: {len(records)}")

    selected_episode_ids = sorted({record.episode_index for record in records})
    log_progress(
        f"Loaded {len(records)} cleaned gold records across {len(selected_episode_ids)} episodes "
        f"and {len({record.task_index for record in records})} tasks."
    )

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=selected_episode_ids,
        download_videos=False,
    )
    resolve_dataset_indices(dataset, records)

    image_keys = list(dataset.meta.image_keys)
    if not image_keys:
        raise RuntimeError("Dataset does not expose any image keys.")
    image_key = image_keys[0]
    state_dim = int(dataset.meta.features[OBS_STATE]["shape"][0])
    action_dim = int(dataset.meta.features[ACTION]["shape"][0])

    device = torch.device(args.device)
    config = build_config(str(device), args.dtype, image_key=image_key, state_dim=state_dim, action_dim=action_dim)
    state_normalizer = build_state_normalizer(config, dataset)
    tokenizer = AutoTokenizer.from_pretrained(resolve_local_hf_snapshot(args.tokenizer), local_files_only=True)
    policy = PI05WordPolicy.from_pretrained(
        resolve_local_hf_snapshot(args.checkpoint),
        config=config,
        local_files_only=True,
        strict=False,
    ).to(device)
    policy.eval()
    policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    features, labels, metadata = extract_hidden_features(
        policy=policy,
        tokenizer=tokenizer,
        dataset=dataset,
        records=records,
        image_key=image_key,
        state_normalizer=state_normalizer,
        batch_size=args.feature_batch_size,
        device=device,
    )
    log_progress(f"Extracted hidden features with shape {tuple(features.shape)}.")

    train_indices, val_indices, split_summary = split_indices_by_target(records, val_ratio=args.val_ratio, seed=args.seed)
    if not val_indices:
        raise RuntimeError("Validation split is empty after target-aware splitting.")

    probe_result = train_probe(
        features=features,
        labels=labels,
        train_indices=train_indices,
        val_indices=val_indices,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.probe_batch_size,
    )

    summary = {
        "annotation_manifest": str(annotation_manifest),
        "dataset_root": args.dataset_root,
        "dataset_repo_id": args.dataset_repo_id,
        "image_key": image_key,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
        "device": str(device),
        "dtype": args.dtype,
        "selection_summary": summarize_selection(records),
        "feature_shape": list(features.shape),
        "split_strategy": "target_aware",
        "split_summary": split_summary,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "probe_result": probe_result,
        "examples": [
            {
                **metadata[idx],
                "primitive": labels[idx].primitive,
                "target": labels[idx].target,
                "relation": labels[idx].relation,
            }
            for idx in range(min(10, len(metadata)))
        ],
    }

    output = json.dumps(summary, indent=2, ensure_ascii=True)
    print(output)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
