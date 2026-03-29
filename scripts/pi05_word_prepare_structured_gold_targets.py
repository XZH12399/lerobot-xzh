#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pi05_word_structured_target_utils import parse_structured_target


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach structured target fields to the cleaned pi05_word gold set.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence_structured.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence_structured_summary.json"),
    )
    args = parser.parse_args()

    records = []
    for line in args.input_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        structured_target = parse_structured_target(row["annotation"]["target"]).as_dict()
        row["structured_target"] = structured_target
        records.append(row)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "num_records": len(records),
        "unique_full_targets": len({row["structured_target"]["full"] for row in records}),
        "unique_target_cores": len({row["structured_target"]["core"] for row in records}),
        "unique_descriptors": len({row["structured_target"]["descriptor"] for row in records}),
        "unique_reference_relations": len({row["structured_target"]["reference_relation"] for row in records}),
        "unique_reference_anchors": len({row["structured_target"]["reference_anchor"] for row in records}),
        "core_counts": dict(sorted(Counter(row["structured_target"]["core"] for row in records).items())),
        "descriptor_counts": dict(sorted(Counter(row["structured_target"]["descriptor"] for row in records).items())),
        "reference_relation_counts": dict(
            sorted(Counter(row["structured_target"]["reference_relation"] for row in records).items())
        ),
        "reference_anchor_counts": dict(
            sorted(Counter(row["structured_target"]["reference_anchor"] for row in records).items())
        ),
        "examples": [
            {
                "sample_id": row["sample_id"],
                "target": row["annotation"]["target"],
                "structured_target": row["structured_target"],
            }
            for row in records[:12]
        ],
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
