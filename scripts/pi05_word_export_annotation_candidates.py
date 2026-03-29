#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def log_progress(message: str) -> None:
    print(f"[pi05_word_annotation] {message}", file=sys.stderr, flush=True)


def scalarize(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


@dataclass(frozen=True)
class CandidatePhase:
    semantic_family: str
    phase_hint: str
    relative_progress: float


@dataclass(frozen=True)
class EpisodeSelection:
    episode_id: int
    task_index: int
    task: str
    task_family: str
    dataset_from_index: int
    dataset_to_index: int


@dataclass(frozen=True)
class CandidateRecord:
    sample_id: str
    semantic_family: str
    phase_hint: str
    task_family: str
    task: str
    task_index: int
    episode_id: int
    absolute_index: int
    relative_progress: float
    dataset_from_index: int
    dataset_to_index: int


PHASES_BY_TASK_FAMILY = {
    "pick_place": [
        CandidatePhase(semantic_family="pick", phase_hint="pregrasp", relative_progress=0.30),
        CandidatePhase(semantic_family="place", phase_hint="transport_or_place", relative_progress=0.72),
    ],
    "place_only": [
        CandidatePhase(semantic_family="place", phase_hint="transport_or_place", relative_progress=0.68),
    ],
    "open_only": [
        CandidatePhase(semantic_family="open", phase_hint="handle_or_open", relative_progress=0.45),
    ],
}


def classify_task(task: str) -> str | None:
    normalized = task.strip().lower()
    if normalized.startswith("pick up "):
        return "pick_place"
    if normalized.startswith("open "):
        return "open_only"
    if normalized.startswith("put "):
        return "place_only"
    return None


def round_robin_select(episodes_by_task: dict[int, list[EpisodeSelection]], max_episodes: int, rng: random.Random) -> list[EpisodeSelection]:
    if max_episodes <= 0:
        return []

    task_ids = list(episodes_by_task)
    rng.shuffle(task_ids)
    for task_id in task_ids:
        rng.shuffle(episodes_by_task[task_id])

    selected: list[EpisodeSelection] = []
    cursor = 0
    while len(selected) < max_episodes:
        progressed = False
        for task_id in task_ids:
            task_episodes = episodes_by_task[task_id]
            if cursor < len(task_episodes):
                selected.append(task_episodes[cursor])
                progressed = True
                if len(selected) >= max_episodes:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def select_episodes(
    meta: LeRobotDatasetMetadata,
    pick_place_episodes: int,
    place_only_episodes: int,
    open_only_episodes: int,
    seed: int,
) -> list[EpisodeSelection]:
    rng = random.Random(seed)
    requested_counts = {
        "pick_place": pick_place_episodes,
        "place_only": place_only_episodes,
        "open_only": open_only_episodes,
    }
    grouped: dict[str, dict[int, list[EpisodeSelection]]] = {
        family: defaultdict(list) for family in requested_counts
    }

    for episode in meta.episodes:
        tasks = episode["tasks"] or []
        if not tasks:
            continue
        task = tasks[0]
        task_family = classify_task(task)
        if task_family not in grouped:
            continue
        task_index = int(meta.tasks.loc[task, "task_index"])
        grouped[task_family][task_index].append(
            EpisodeSelection(
                episode_id=int(episode["episode_index"]),
                task_index=task_index,
                task=task,
                task_family=task_family,
                dataset_from_index=int(episode["dataset_from_index"]),
                dataset_to_index=int(episode["dataset_to_index"]),
            )
        )

    selected: list[EpisodeSelection] = []
    for task_family, max_episodes in requested_counts.items():
        family_selected = round_robin_select(grouped[task_family], max_episodes=max_episodes, rng=rng)
        selected.extend(family_selected)
        log_progress(
            f"Selected {len(family_selected)} episodes for {task_family} "
            f"(requested {max_episodes}, available {sum(len(v) for v in grouped[task_family].values())})."
        )

    return sorted(selected, key=lambda episode: episode.episode_id)


def build_candidates(selected_episodes: list[EpisodeSelection]) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    for episode in selected_episodes:
        episode_length = episode.dataset_to_index - episode.dataset_from_index
        if episode_length <= 0:
            continue
        for phase in PHASES_BY_TASK_FAMILY[episode.task_family]:
            offset = int(round((episode_length - 1) * phase.relative_progress))
            absolute_index = episode.dataset_from_index + max(0, min(episode_length - 1, offset))
            sample_id = f"{phase.semantic_family}_ep{episode.episode_id:04d}_abs{absolute_index:06d}"
            candidates.append(
                CandidateRecord(
                    sample_id=sample_id,
                    semantic_family=phase.semantic_family,
                    phase_hint=phase.phase_hint,
                    task_family=episode.task_family,
                    task=episode.task,
                    task_index=episode.task_index,
                    episode_id=episode.episode_id,
                    absolute_index=absolute_index,
                    relative_progress=phase.relative_progress,
                    dataset_from_index=episode.dataset_from_index,
                    dataset_to_index=episode.dataset_to_index,
                )
            )
    return candidates


def tensor_to_pil(image_tensor) -> Image.Image:
    image_np = (
        image_tensor.detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(image_np)


def make_context_strip(images: list[Image.Image], gap: int = 8) -> Image.Image:
    width, height = images[0].size
    canvas = Image.new("RGB", (width * len(images) + gap * (len(images) - 1), height), color=(255, 255, 255))
    for idx, image in enumerate(images):
        x_offset = idx * (width + gap)
        canvas.paste(image, (x_offset, 0))
    return canvas


def export_candidates(
    dataset: LeRobotDataset,
    candidates: list[CandidateRecord],
    output_dir: Path,
    context_stride: int,
) -> tuple[list[dict], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    context_dir = output_dir / "context"
    image_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)

    image_key = list(dataset.meta.image_keys)[0]
    records: list[dict] = []

    for candidate in candidates:
        dataset_index = dataset._absolute_to_relative_idx.get(candidate.absolute_index)
        if dataset_index is None:
            raise RuntimeError(f"Missing dataset index for absolute frame {candidate.absolute_index}")

        item = dataset[int(dataset_index)]
        current_image = tensor_to_pil(item[image_key])
        image_path = image_dir / f"{candidate.sample_id}.png"
        current_image.save(image_path)

        context_images = []
        context_absolute_indices = []
        for delta in (-context_stride, 0, context_stride):
            absolute_index = candidate.absolute_index + delta
            absolute_index = max(candidate.dataset_from_index, min(candidate.dataset_to_index - 1, absolute_index))
            context_dataset_index = dataset._absolute_to_relative_idx.get(absolute_index)
            if context_dataset_index is None:
                context_dataset_index = dataset_index
                absolute_index = candidate.absolute_index
            context_item = dataset[int(context_dataset_index)]
            context_images.append(tensor_to_pil(context_item[image_key]))
            context_absolute_indices.append(int(absolute_index))

        context_strip = make_context_strip(context_images)
        context_path = context_dir / f"{candidate.sample_id}_context.png"
        context_strip.save(context_path)

        record = {
            "sample_id": candidate.sample_id,
            "semantic_family": candidate.semantic_family,
            "phase_hint": candidate.phase_hint,
            "task_family": candidate.task_family,
            "instruction": candidate.task,
            "task_index": candidate.task_index,
            "episode_id": candidate.episode_id,
            "absolute_index": candidate.absolute_index,
            "frame_index": int(scalarize(item["frame_index"])),
            "relative_progress": candidate.relative_progress,
            "image_key": image_key,
            "image_path": str(image_path),
            "context_image_path": str(context_path),
            "context_absolute_indices": context_absolute_indices,
            "annotation": {
                "primitive": None,
                "target": None,
                "relation": None,
                "confidence": None,
                "note": None,
            },
        }
        records.append(record)

    summary = {
        "num_candidates": len(records),
        "semantic_family_counts": dict(sorted(Counter(record["semantic_family"] for record in records).items())),
        "task_family_counts": dict(sorted(Counter(record["task_family"] for record in records).items())),
        "unique_tasks": len({record["instruction"] for record in records}),
        "unique_episodes": len({record["episode_id"] for record in records}),
        "includes_wipe": False,
        "note": "Current LIBERO subset exposes pick/place/open tasks but no wipe tasks.",
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export round-1 pi05_word semantics annotation candidates.")
    parser.add_argument("--dataset-root", default="/home/ct_24210860031/data/libero")
    parser.add_argument("--dataset-repo-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--pick-place-episodes", type=int, default=15)
    parser.add_argument("--place-only-episodes", type=int, default=10)
    parser.add_argument("--open-only-episodes", type=int, default=10)
    parser.add_argument("--context-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1"),
    )
    args = parser.parse_args()

    meta = LeRobotDatasetMetadata(repo_id=args.dataset_repo_id, root=args.dataset_root)
    selected_episodes = select_episodes(
        meta=meta,
        pick_place_episodes=args.pick_place_episodes,
        place_only_episodes=args.place_only_episodes,
        open_only_episodes=args.open_only_episodes,
        seed=args.seed,
    )
    if not selected_episodes:
        raise RuntimeError("No candidate episodes selected.")

    candidates = build_candidates(selected_episodes)
    selected_episode_ids = sorted({candidate.episode_id for candidate in candidates})
    log_progress(
        f"Loading filtered dataset for {len(selected_episode_ids)} episodes "
        f"to export {len(candidates)} annotation candidates."
    )
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=selected_episode_ids,
        download_videos=False,
    )

    records, summary = export_candidates(
        dataset=dataset,
        candidates=candidates,
        output_dir=args.output_dir,
        context_stride=args.context_stride,
    )

    manifest_path = args.output_dir / "candidates.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(json.dumps({"manifest_path": str(manifest_path), "summary_path": str(summary_path), **summary}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
