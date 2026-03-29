#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi05_word.configuration_pi05_word import PI05WordConfig
from lerobot.policies.pi05_word.modeling_pi05_word import PI05WordPolicy, make_att_2d_masks
from lerobot.processor import NormalizerProcessorStep, TransitionKey
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

RELATION_NONE = "none"
OBJECT_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "back",
    "big",
    "black",
    "blue",
    "bottom",
    "brown",
    "center",
    "door",
    "front",
    "green",
    "handle",
    "in",
    "inside",
    "into",
    "left",
    "lid",
    "little",
    "middle",
    "of",
    "on",
    "onto",
    "open",
    "pick",
    "place",
    "put",
    "red",
    "right",
    "small",
    "the",
    "to",
    "top",
    "up",
    "upper",
    "white",
    "with",
    "wooden",
    "yellow",
}
OBJECT_ALIASES = {
    "bowls": "bowl",
    "cups": "cup",
    "mugs": "mug",
    "plates": "plate",
    "drawers": "drawer",
    "cabinets": "cabinet",
    "blocks": "block",
    "buttons": "button",
    "doors": "door",
    "handles": "handle",
    "microwaves": "microwave",
    "stoves": "stove",
    "burners": "burner",
    "pots": "pot",
    "pans": "pan",
    "shelves": "shelf",
    "racks": "rack",
    "tables": "table",
    "trays": "tray",
    "lids": "lid",
    "books": "book",
}
OBJECT_PHRASE_ALIASES = {
    "orange juice": "orange_juice",
    "cream cheese": "cream_cheese",
    "alphabet soup": "alphabet_soup",
    "bbq sauce": "bbq_sauce",
    "salad dressing": "salad_dressing",
    "tomato sauce": "tomato_sauce",
    "chocolate pudding": "chocolate_pudding",
    "wine bottle": "wine_bottle",
    "cream cheese box": "cream_cheese_box",
    "cookie box": "cookie_box",
    "black bowl": "bowl",
    "white mug": "mug",
    "yellow and white mug": "mug",
    "moka pot": "moka_pot",
}
CLAUSE_SPLIT_MARKERS = [
    " and put ",
    " and place ",
    " and close ",
    " and open ",
    " and pick ",
    " and stack ",
    " then ",
]
PREPOSITION_RELATIONS = [
    (" next to ", "near"),
    (" near ", "near"),
    (" inside ", "inside"),
    (" into ", "inside"),
    (" in ", "inside"),
    (" on top of ", "on"),
    (" onto ", "on"),
    (" on ", "on"),
]
PICK_LOCATION_MARKERS = [
    " next to ",
    " on ",
    " in ",
    " inside ",
    " from ",
    " between ",
]


@dataclass(frozen=True)
class SingleStepLabel:
    primitive: str
    target: str
    relation: str


@dataclass
class SampleRecord:
    dataset_index: int
    absolute_index: int
    episode_index: int
    task_index: int
    task: str
    label: SingleStepLabel


class SingleStepProbe(nn.Module):
    def __init__(self, input_dim: int, num_primitives: int, num_targets: int, num_relations: int):
        super().__init__()
        hidden_dim = min(512, max(128, input_dim // 4))
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.primitive_head = nn.Linear(hidden_dim, num_primitives)
        self.target_head = nn.Linear(hidden_dim, num_targets)
        self.relation_head = nn.Linear(hidden_dim, num_relations)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(x)
        return {
            "primitive_logits": self.primitive_head(hidden),
            "target_logits": self.target_head(hidden),
            "relation_logits": self.relation_head(hidden),
        }


def resolve_local_hf_snapshot(model_id_or_path: str) -> str:
    path = Path(model_id_or_path)
    if path.exists():
        return str(path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_dir = cache_root / f"models--{model_id_or_path.replace('/', '--')}"
    if not cache_dir.exists():
        return model_id_or_path

    ref_file = cache_dir / "refs" / "main"
    if ref_file.exists():
        commit = ref_file.read_text(encoding="utf-8").strip()
        snapshot_dir = cache_dir / "snapshots" / commit
        if snapshot_dir.exists():
            return str(snapshot_dir)

    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if snapshots:
            return str(snapshots[0])

    return model_id_or_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalarize(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value
    return value


def log_progress(message: str) -> None:
    print(f"[pi05_word_probe] {message}", file=sys.stderr, flush=True)


def canonicalize_object(phrase: str) -> str:
    lower_phrase = phrase.lower()
    for phrase_key, canonical in OBJECT_PHRASE_ALIASES.items():
        if phrase_key in lower_phrase:
            return canonical

    tokens = re.findall(r"[a-z]+", phrase.lower())
    filtered = []
    for token in tokens:
        token = OBJECT_ALIASES.get(token, token)
        if token in OBJECT_STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and token not in OBJECT_ALIASES:
            token = token[:-1]
        filtered.append(token)
    if not filtered:
        return "none"
    for preferred in ["drawer", "door", "microwave", "basket", "plate", "stove", "cabinet", "rack"]:
        if preferred in filtered:
            return preferred
    return filtered[-1]


def select_relevant_clause(task: str) -> str:
    clause = " ".join(task.lower().replace("_", " ").split())
    for marker in CLAUSE_SPLIT_MARKERS:
        if marker in clause:
            return clause.split(marker, 1)[0]
    return clause


def infer_single_step_label(task: str) -> SingleStepLabel | None:
    normalized = select_relevant_clause(task)

    pick_match = re.search(r"(pick up|pick|grasp|lift|take)\s+(.*)", normalized)
    if pick_match:
        target_phrase = pick_match.group(2)
        for marker in PICK_LOCATION_MARKERS:
            if marker in target_phrase:
                target_phrase = target_phrase.split(marker, 1)[0]
                break
        return SingleStepLabel(primitive="approach", target=canonicalize_object(target_phrase), relation="above")

    wipe_match = re.search(r"(wipe|clean)\s+(.*)", normalized)
    if wipe_match:
        target_phrase = wipe_match.group(2)
        return SingleStepLabel(
            primitive="follow_path", target=canonicalize_object(target_phrase), relation="contact"
        )

    open_match = re.search(r"open\s+(.*)", normalized)
    if open_match:
        target_phrase = open_match.group(1)
        return SingleStepLabel(primitive="open", target=canonicalize_object(target_phrase), relation=RELATION_NONE)

    close_match = re.search(r"close\s+(.*)", normalized)
    if close_match:
        target_phrase = close_match.group(1)
        return SingleStepLabel(primitive="close", target=canonicalize_object(target_phrase), relation=RELATION_NONE)

    for verb in ["put", "place", "insert", "stack"]:
        if normalized.startswith(f"{verb} ") or f" {verb} " in normalized:
            for needle, relation in PREPOSITION_RELATIONS:
                if needle in normalized:
                    target_phrase = normalized.split(needle, 1)[1]
                    return SingleStepLabel(
                        primitive="move_to",
                        target=canonicalize_object(target_phrase),
                        relation=relation,
                    )
            tail = normalized.split(verb, 1)[1]
            return SingleStepLabel(primitive="move_to", target=canonicalize_object(tail), relation="near")

    return None


def build_config(device: str, dtype: str, image_key: str, state_dim: int, action_dim: int) -> PI05WordConfig:
    config = PI05WordConfig(device=device, dtype=dtype)
    config.input_features = {
        image_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *config.image_resolution)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}
    return config


def build_state_normalizer(config: PI05WordConfig, dataset: LeRobotDataset) -> NormalizerProcessorStep:
    return NormalizerProcessorStep(
        features={OBS_STATE: config.input_features[OBS_STATE]},
        norm_map=config.normalization_mapping,
        stats=dataset.meta.stats,
        device="cpu",
    )


def build_prompt(task: str, state: torch.Tensor, state_normalizer: NormalizerProcessorStep) -> str:
    transition = {TransitionKey.OBSERVATION: {OBS_STATE: state}}
    normalized = state_normalizer(transition)[TransitionKey.OBSERVATION][OBS_STATE]
    state_np = normalized.detach().cpu().numpy()
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_states.tolist()))
    cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned_text}, State: {state_str};\nAction: "


def collect_supported_tasks(tasks_df) -> dict[int, tuple[str, SingleStepLabel]]:
    supported: dict[int, tuple[str, SingleStepLabel]] = {}
    for task_name, row in tasks_df.iterrows():
        label = infer_single_step_label(task_name)
        if label is not None:
            supported[int(row.task_index)] = (task_name, label)
    return supported


def select_samples_from_metadata(
    meta: LeRobotDatasetMetadata,
    supported_tasks: dict[int, tuple[str, SingleStepLabel]],
    max_samples: int,
    max_tasks: int,
    max_per_task: int,
    seed: int,
) -> list[SampleRecord]:
    rng = random.Random(seed)
    task_name_to_index = {task_name: task_idx for task_idx, (task_name, _) in supported_tasks.items()}

    grouped: dict[int, list[SampleRecord]] = defaultdict(list)
    for episode in meta.episodes:
        episode_id = int(episode["episode_index"])
        dataset_from_index = int(episode["dataset_from_index"])
        dataset_to_index = int(episode["dataset_to_index"])
        if dataset_to_index <= dataset_from_index:
            continue

        task_names = episode["tasks"] or []
        matched_task_name = next((task_name for task_name in task_names if task_name in task_name_to_index), None)
        if matched_task_name is None:
            continue

        task_idx = task_name_to_index[matched_task_name]
        _, label = supported_tasks[task_idx]
        midpoint_absolute_index = (dataset_from_index + dataset_to_index - 1) // 2
        grouped[task_idx].append(
            SampleRecord(
                dataset_index=-1,
                absolute_index=midpoint_absolute_index,
                episode_index=episode_id,
                task_index=task_idx,
                task=matched_task_name,
                label=label,
            )
        )

    ordered_task_ids = [task_idx for task_idx, task_records in grouped.items() if task_records]
    rng.shuffle(ordered_task_ids)
    ordered_task_ids.sort(key=lambda task_idx: len(grouped[task_idx]), reverse=True)
    if max_tasks > 0:
        ordered_task_ids = ordered_task_ids[:max_tasks]

    per_task_records: dict[int, list[SampleRecord]] = {}
    for task_idx in ordered_task_ids:
        task_records = list(grouped[task_idx])
        rng.shuffle(task_records)
        per_task_records[task_idx] = task_records[:max_per_task]

    selected: list[SampleRecord] = []
    cursor = 0
    while len(selected) < max_samples:
        progressed = False
        for task_idx in ordered_task_ids:
            task_records = per_task_records[task_idx]
            if cursor < len(task_records):
                selected.append(task_records[cursor])
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        cursor += 1

    return selected


def resolve_dataset_indices(dataset: LeRobotDataset, records: list[SampleRecord]) -> None:
    missing = []
    absolute_to_relative = dataset._absolute_to_relative_idx
    for record in records:
        dataset_index = absolute_to_relative.get(record.absolute_index)
        if dataset_index is None:
            missing.append(record.absolute_index)
            continue
        record.dataset_index = int(dataset_index)

    if missing:
        preview = ", ".join(map(str, missing[:5]))
        raise RuntimeError(f"Failed to map absolute frame indices into filtered dataset: {preview}")


def print_task_inspection(supported_tasks: dict[int, tuple[str, SingleStepLabel]]) -> None:
    summary = []
    for task_idx, (task_name, label) in sorted(supported_tasks.items()):
        summary.append(
            {
                "task_index": task_idx,
                "task": task_name,
                "primitive": label.primitive,
                "target": label.target,
                "relation": label.relation,
            }
        )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def extract_hidden_features(
    policy: PI05WordPolicy,
    tokenizer: AutoTokenizer,
    dataset: LeRobotDataset,
    records: list[SampleRecord],
    image_key: str,
    state_normalizer: NormalizerProcessorStep,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[SingleStepLabel], list[dict[str, int | str]]]:
    features = []
    labels = []
    metadata = []

    for batch_start in range(0, len(records), batch_size):
        batch_records = records[batch_start : batch_start + batch_size]
        images = []
        prompts = []
        for record in batch_records:
            item = dataset[record.dataset_index]
            images.append(item[image_key])
            prompts.append(build_prompt(record.task, item[OBS_STATE], state_normalizer))
            labels.append(record.label)
            metadata.append(
                {
                    "dataset_index": record.dataset_index,
                    "absolute_index": record.absolute_index,
                    "episode_index": record.episode_index,
                    "task_index": record.task_index,
                    "task": record.task,
                }
            )

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=policy.config.tokenizer_max_length,
        )
        batch = {
            image_key: torch.stack(images).to(device=device, dtype=torch.float32),
            OBS_LANGUAGE_TOKENS: tokenized["input_ids"].to(device),
            OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].to(device=device, dtype=torch.bool),
        }

        with torch.inference_mode():
            image_list, img_masks = policy._preprocess_images(batch)
            tokens = batch[OBS_LANGUAGE_TOKENS]
            masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                image_list, img_masks, tokens, masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = policy.model._prepare_attention_masks_4d(prefix_att_2d_masks)
            (prefix_hidden, _), _ = policy.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            hidden_mask = prefix_pad_masks.to(prefix_hidden.dtype).unsqueeze(-1)
            pooled_hidden = (prefix_hidden * hidden_mask).sum(dim=1) / hidden_mask.sum(dim=1).clamp_min(1.0)
            features.append(pooled_hidden.detach().cpu())

    return torch.cat(features, dim=0), labels, metadata


def encode_labels(labels: list[str]) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    classes = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(classes)}
    encoded = torch.tensor([mapping[label] for label in labels], dtype=torch.long)
    return encoded, classes, mapping


def split_records(records: list[SampleRecord], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    task_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        task_to_indices[record.task_index].append(idx)

    train_indices = []
    val_indices = []
    for _, indices in task_to_indices.items():
        rng.shuffle(indices)
        if len(indices) >= 4:
            val_count = max(1, int(round(len(indices) * val_ratio)))
        else:
            val_count = 0
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    return sorted(train_indices), sorted(val_indices)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == targets).float().mean().item())


def majority_baseline(train_targets: torch.Tensor, eval_targets: torch.Tensor) -> float:
    counts = Counter(train_targets.tolist())
    majority = counts.most_common(1)[0][0]
    return float((eval_targets == majority).float().mean().item())


def train_probe(
    features: torch.Tensor,
    labels: list[SingleStepLabel],
    train_indices: list[int],
    val_indices: list[int],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict:
    features = features.float()
    primitive_targets, primitive_classes, _ = encode_labels([label.primitive for label in labels])
    target_targets, target_classes, _ = encode_labels([label.target for label in labels])
    relation_targets, relation_classes, _ = encode_labels([label.relation for label in labels])

    probe = SingleStepProbe(
        input_dim=features.shape[-1],
        num_primitives=len(primitive_classes),
        num_targets=len(target_classes),
        num_relations=len(relation_classes),
    ).to(device)

    train_dataset = TensorDataset(
        features[train_indices],
        primitive_targets[train_indices],
        target_targets[train_indices],
        relation_targets[train_indices],
    )
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    best_state = None
    best_val_exact = -math.inf
    history = []

    if val_indices:
        val_features = features[val_indices].to(device)
        val_primitive = primitive_targets[val_indices].to(device)
        val_target = target_targets[val_indices].to(device)
        val_relation = relation_targets[val_indices].to(device)
    else:
        val_features = val_primitive = val_target = val_relation = None

    for epoch in range(1, epochs + 1):
        probe.train()
        epoch_loss = 0.0
        sample_count = 0
        for batch_features, batch_primitive, batch_target, batch_relation in train_loader:
            batch_features = batch_features.to(device)
            batch_primitive = batch_primitive.to(device)
            batch_target = batch_target.to(device)
            batch_relation = batch_relation.to(device)

            outputs = probe(batch_features)
            loss = (
                F.cross_entropy(outputs["primitive_logits"], batch_primitive)
                + F.cross_entropy(outputs["target_logits"], batch_target)
                + F.cross_entropy(outputs["relation_logits"], batch_relation)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_features.shape[0]
            epoch_loss += loss.item() * batch_size_actual
            sample_count += batch_size_actual

        train_loss = epoch_loss / max(1, sample_count)
        metrics = {"epoch": epoch, "train_loss": train_loss}

        if val_indices:
            probe.eval()
            with torch.inference_mode():
                outputs = probe(val_features)
                primitive_acc = accuracy(outputs["primitive_logits"], val_primitive)
                target_acc = accuracy(outputs["target_logits"], val_target)
                relation_acc = accuracy(outputs["relation_logits"], val_relation)
                exact = float(
                    (
                        (outputs["primitive_logits"].argmax(dim=-1) == val_primitive)
                        & (outputs["target_logits"].argmax(dim=-1) == val_target)
                        & (outputs["relation_logits"].argmax(dim=-1) == val_relation)
                    )
                    .float()
                    .mean()
                    .item()
                )
                metrics.update(
                    {
                        "val_primitive_acc": primitive_acc,
                        "val_target_acc": target_acc,
                        "val_relation_acc": relation_acc,
                        "val_exact_match": exact,
                    }
                )
                if exact > best_val_exact:
                    best_val_exact = exact
                    best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

        history.append(metrics)

    if best_state is not None:
        probe.load_state_dict(best_state)

    result = {
        "history": history,
        "primitive_classes": primitive_classes,
        "target_classes": target_classes,
        "relation_classes": relation_classes,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
    }

    if val_indices:
        probe.eval()
        with torch.inference_mode():
            outputs = probe(val_features)
            result["final_metrics"] = {
                "primitive_acc": accuracy(outputs["primitive_logits"], val_primitive),
                "target_acc": accuracy(outputs["target_logits"], val_target),
                "relation_acc": accuracy(outputs["relation_logits"], val_relation),
                "exact_match": float(
                    (
                        (outputs["primitive_logits"].argmax(dim=-1) == val_primitive)
                        & (outputs["target_logits"].argmax(dim=-1) == val_target)
                        & (outputs["relation_logits"].argmax(dim=-1) == val_relation)
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "primitive_majority_baseline": majority_baseline(primitive_targets[train_indices], val_primitive.cpu()),
                "target_majority_baseline": majority_baseline(target_targets[train_indices], val_target.cpu()),
                "relation_majority_baseline": majority_baseline(relation_targets[train_indices], val_relation.cpu()),
            }

    return result


def summarize_selection(records: list[SampleRecord]) -> dict:
    primitive_counts = Counter(record.label.primitive for record in records)
    target_counts = Counter(record.label.target for record in records)
    relation_counts = Counter(record.label.relation for record in records)
    task_counts = Counter(record.task for record in records)
    episode_counts = Counter(record.episode_index for record in records)
    return {
        "num_records": len(records),
        "num_unique_tasks": len(task_counts),
        "num_unique_episodes": len(episode_counts),
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Real-sample single-step probe for pi05_word.")
    parser.add_argument("--dataset-root", default="/home/ct_24210860031/data/libero")
    parser.add_argument("--dataset-repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--checkpoint", default="lerobot/pi05_libero_base")
    parser.add_argument("--tokenizer", default="google/paligemma-3b-pt-224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--max-samples", type=int, default=36)
    parser.add_argument("--max-tasks", type=int, default=9)
    parser.add_argument("--max-per-task", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=6)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    meta = LeRobotDatasetMetadata(repo_id=args.dataset_repo_id, root=args.dataset_root)
    supported_tasks = collect_supported_tasks(meta.tasks)

    if args.inspect_only:
        print_task_inspection(supported_tasks)
        return

    if not supported_tasks:
        raise RuntimeError("No supported tasks found for heuristic single-step labeling.")

    records = select_samples_from_metadata(
        meta=meta,
        supported_tasks=supported_tasks,
        max_samples=args.max_samples,
        max_tasks=args.max_tasks,
        max_per_task=args.max_per_task,
        seed=args.seed,
    )
    if len(records) < 8:
        raise RuntimeError(f"Not enough supported samples selected: {len(records)}")

    selected_episode_ids = sorted({record.episode_index for record in records})
    log_progress(
        "Selected "
        f"{len(records)} samples across {len(selected_episode_ids)} episodes and "
        f"{len({record.task_index for record in records})} tasks from metadata."
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
    log_progress(f"Loaded filtered dataset with {len(dataset)} frames using image key '{image_key}'.")

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

    train_indices, val_indices = split_records(records, val_ratio=args.val_ratio, seed=args.seed)
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
        "selected_episode_ids": selected_episode_ids,
        "feature_shape": list(features.shape),
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
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
