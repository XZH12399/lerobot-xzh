#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
for path in [SCRIPT_ROOT, SRC_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_word.modeling_pi05_word import PI05WordPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from pi05_word_single_step_probe import (
    SampleRecord,
    SingleStepLabel,
    accuracy,
    build_config,
    build_state_normalizer,
    encode_labels,
    extract_hidden_features,
    majority_baseline,
    resolve_dataset_indices,
    resolve_local_hf_snapshot,
    set_seed,
    summarize_selection,
)


HEAD_ORDER = [
    "primitive",
    "relation",
    "target_core",
    "target_descriptor",
    "target_reference_relation",
    "target_reference_anchor",
]


class StructuredTargetProbe(nn.Module):
    def __init__(self, input_dim: int, head_dims: dict[str, int]):
        super().__init__()
        hidden_dim = min(512, max(128, input_dim // 4))
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, size) for name, size in head_dims.items()})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(x)
        return {name: head(hidden) for name, head in self.heads.items()}


def log_progress(message: str) -> None:
    print(f"[pi05_word_structured_probe] {message}", file=sys.stderr, flush=True)


def load_structured_records(annotation_manifest: Path) -> tuple[list[SampleRecord], list[dict[str, str]]]:
    records: list[SampleRecord] = []
    structured_targets: list[dict[str, str]] = []
    for line in annotation_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        annotation = row["annotation"]
        structured_target = row["structured_target"]
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
        structured_targets.append(
            {
                "target_core": str(structured_target["core"]),
                "target_descriptor": str(structured_target["descriptor"]),
                "target_reference_relation": str(structured_target["reference_relation"]),
                "target_reference_anchor": str(structured_target["reference_anchor"]),
            }
        )
    return records, structured_targets


def split_indices_by_core(
    records: list[SampleRecord],
    structured_targets: list[dict[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], dict]:
    rng = random.Random(seed)
    core_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, structured in enumerate(structured_targets):
        core_to_indices[structured["target_core"]].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    split_summary = {"singleton_cores_kept_in_train": 0, "cores_with_val": 0}

    for core, indices in sorted(core_to_indices.items()):
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            split_summary["singleton_cores_kept_in_train"] += 1
            continue
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
        split_summary["cores_with_val"] += 1

    return sorted(train_indices), sorted(val_indices), split_summary


def train_structured_probe(
    features: torch.Tensor,
    records: list[SampleRecord],
    structured_targets: list[dict[str, str]],
    train_indices: list[int],
    val_indices: list[int],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict:
    features = features.float()
    slot_labels = {
        "primitive": [record.label.primitive for record in records],
        "relation": [record.label.relation for record in records],
        "target_core": [target["target_core"] for target in structured_targets],
        "target_descriptor": [target["target_descriptor"] for target in structured_targets],
        "target_reference_relation": [target["target_reference_relation"] for target in structured_targets],
        "target_reference_anchor": [target["target_reference_anchor"] for target in structured_targets],
    }

    encoded_targets: dict[str, torch.Tensor] = {}
    class_names: dict[str, list[str]] = {}
    for name, labels in slot_labels.items():
        encoded, classes, _ = encode_labels(labels)
        encoded_targets[name] = encoded
        class_names[name] = classes

    probe = StructuredTargetProbe(
        input_dim=features.shape[-1],
        head_dims={name: len(class_names[name]) for name in HEAD_ORDER},
    ).to(device)

    train_dataset = TensorDataset(features[train_indices], *[encoded_targets[name][train_indices] for name in HEAD_ORDER])
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val_semantics_exact = float("-inf")
    history = []

    val_features = features[val_indices].to(device)
    val_targets = {name: encoded_targets[name][val_indices].to(device) for name in HEAD_ORDER}

    for epoch in range(1, epochs + 1):
        probe.train()
        epoch_loss = 0.0
        sample_count = 0
        for batch in train_loader:
            batch_features = batch[0].to(device)
            batch_targets = {name: tensor.to(device) for name, tensor in zip(HEAD_ORDER, batch[1:], strict=True)}
            outputs = probe(batch_features)
            loss = sum(F.cross_entropy(outputs[name], batch_targets[name]) for name in HEAD_ORDER)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_features.shape[0]
            epoch_loss += loss.item() * batch_size_actual
            sample_count += batch_size_actual

        metrics = {"epoch": epoch, "train_loss": epoch_loss / max(1, sample_count)}

        probe.eval()
        with torch.inference_mode():
            outputs = probe(val_features)
            for name in HEAD_ORDER:
                metrics[f"val_{name}_acc"] = accuracy(outputs[name], val_targets[name])

            target_exact = (
                (outputs["target_core"].argmax(dim=-1) == val_targets["target_core"])
                & (outputs["target_descriptor"].argmax(dim=-1) == val_targets["target_descriptor"])
                & (
                    outputs["target_reference_relation"].argmax(dim=-1)
                    == val_targets["target_reference_relation"]
                )
                & (outputs["target_reference_anchor"].argmax(dim=-1) == val_targets["target_reference_anchor"])
            )
            semantics_exact = target_exact & (outputs["primitive"].argmax(dim=-1) == val_targets["primitive"]) & (
                outputs["relation"].argmax(dim=-1) == val_targets["relation"]
            )
            metrics["val_target_exact"] = float(target_exact.float().mean().item())
            metrics["val_semantics_exact"] = float(semantics_exact.float().mean().item())

            if metrics["val_semantics_exact"] > best_val_semantics_exact:
                best_val_semantics_exact = metrics["val_semantics_exact"]
                best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}

        history.append(metrics)

    if best_state is not None:
        probe.load_state_dict(best_state)

    result = {
        "history": history,
        "classes": class_names,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
    }

    probe.eval()
    with torch.inference_mode():
        outputs = probe(val_features)
        final_metrics = {f"{name}_acc": accuracy(outputs[name], val_targets[name]) for name in HEAD_ORDER}
        target_exact = (
            (outputs["target_core"].argmax(dim=-1) == val_targets["target_core"])
            & (outputs["target_descriptor"].argmax(dim=-1) == val_targets["target_descriptor"])
            & (outputs["target_reference_relation"].argmax(dim=-1) == val_targets["target_reference_relation"])
            & (outputs["target_reference_anchor"].argmax(dim=-1) == val_targets["target_reference_anchor"])
        )
        semantics_exact = target_exact & (outputs["primitive"].argmax(dim=-1) == val_targets["primitive"]) & (
            outputs["relation"].argmax(dim=-1) == val_targets["relation"]
        )
        final_metrics["target_exact"] = float(target_exact.float().mean().item())
        final_metrics["semantics_exact"] = float(semantics_exact.float().mean().item())

        for name in HEAD_ORDER:
            final_metrics[f"{name}_majority_baseline"] = majority_baseline(
                encoded_targets[name][train_indices], val_targets[name].cpu()
            )

    result["final_metrics"] = final_metrics
    return result


def structured_summary(structured_targets: list[dict[str, str]]) -> dict:
    return {
        "target_core_counts": dict(sorted(Counter(target["target_core"] for target in structured_targets).items())),
        "target_descriptor_counts": dict(
            sorted(Counter(target["target_descriptor"] for target in structured_targets).items())
        ),
        "target_reference_relation_counts": dict(
            sorted(Counter(target["target_reference_relation"] for target in structured_targets).items())
        ),
        "target_reference_anchor_counts": dict(
            sorted(Counter(target["target_reference_anchor"] for target in structured_targets).items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe pi05_word hidden states with structured target slots.")
    parser.add_argument(
        "--annotation-manifest",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence_structured.jsonl"),
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
        default=Path("outputs/pi05_word_validation/pi05_word_structured_target_probe_high_confidence.json"),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    annotation_manifest = args.annotation_manifest.resolve()
    records, structured_targets = load_structured_records(annotation_manifest)
    if len(records) < 8:
        raise RuntimeError(f"Not enough structured gold records found: {len(records)}")

    selected_episode_ids = sorted({record.episode_index for record in records})
    log_progress(
        f"Loaded {len(records)} structured gold records across {len(selected_episode_ids)} episodes "
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

    train_indices, val_indices, split_summary = split_indices_by_core(
        records=records,
        structured_targets=structured_targets,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if not val_indices:
        raise RuntimeError("Validation split is empty after core-aware splitting.")

    probe_result = train_structured_probe(
        features=features,
        records=records,
        structured_targets=structured_targets,
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
        "structured_target_summary": structured_summary(structured_targets),
        "feature_shape": list(features.shape),
        "split_strategy": "core_aware",
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
                **structured_targets[idx],
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
