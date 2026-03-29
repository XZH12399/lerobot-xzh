#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_label(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    return text.lower()


def normalize_target(value: object) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = text.lower().replace("-", "_").replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_.")


def is_complete(annotation: dict) -> bool:
    return all(normalize_text(annotation.get(key)) for key in ["primitive", "target", "relation", "confidence"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a cleaned high-confidence pi05_word gold annotation set.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/candidates.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence_summary.json"),
    )
    parser.add_argument("--confidence", default="high")
    args = parser.parse_args()

    accepted_confidence = {part.strip().lower() for part in args.confidence.split(",") if part.strip()}
    records = []
    for line in args.input_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

    cleaned = []
    dropped_counts = Counter()
    for record in records:
        annotation = record.get("annotation", {})
        if not is_complete(annotation):
            dropped_counts["incomplete"] += 1
            continue

        confidence = normalize_label(annotation.get("confidence"))
        if confidence not in accepted_confidence:
            dropped_counts[f"confidence:{confidence or 'missing'}"] += 1
            continue

        primitive = normalize_label(annotation.get("primitive"))
        target = normalize_target(annotation.get("target"))
        relation = normalize_label(annotation.get("relation"))
        if not primitive or not target or not relation:
            dropped_counts["normalized_empty"] += 1
            continue

        cleaned.append(
            {
                "sample_id": record.get("sample_id"),
                "semantic_family": record.get("semantic_family"),
                "phase_hint": record.get("phase_hint"),
                "task_family": record.get("task_family"),
                "instruction": record.get("instruction"),
                "task_index": int(record.get("task_index")),
                "episode_id": int(record.get("episode_id")),
                "absolute_index": int(record.get("absolute_index")),
                "frame_index": int(record.get("frame_index")),
                "relative_progress": float(record.get("relative_progress")),
                "annotation": {
                    "primitive": primitive,
                    "target": target,
                    "relation": relation,
                    "confidence": confidence,
                    "note": normalize_text(annotation.get("note")),
                },
                "source": {
                    "image_key": record.get("image_key"),
                    "image_path": record.get("image_path"),
                    "context_image_path": record.get("context_image_path"),
                    "context_absolute_indices": record.get("context_absolute_indices", []),
                },
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in cleaned:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "input_manifest": str(args.input_manifest),
        "output_jsonl": str(args.output_jsonl),
        "accepted_confidence": sorted(accepted_confidence),
        "num_input_records": len(records),
        "num_cleaned_records": len(cleaned),
        "dropped_counts": dict(sorted(dropped_counts.items())),
        "primitive_counts": dict(sorted(Counter(record["annotation"]["primitive"] for record in cleaned).items())),
        "target_counts": dict(sorted(Counter(record["annotation"]["target"] for record in cleaned).items())),
        "relation_counts": dict(sorted(Counter(record["annotation"]["relation"] for record in cleaned).items())),
        "task_family_counts": dict(sorted(Counter(record["task_family"] for record in cleaned).items())),
        "unique_tasks": len({record["task_index"] for record in cleaned}),
        "unique_episodes": len({record["episode_id"] for record in cleaned}),
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
