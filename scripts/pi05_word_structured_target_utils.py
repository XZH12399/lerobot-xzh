#!/usr/bin/env python

from __future__ import annotations

from dataclasses import asdict, dataclass

RELATION_PATTERNS = [
    ("next_to", "_next_to_"),
    ("from", "_from_"),
    ("on", "_on_"),
    ("of", "_of_"),
]
GENERIC_SUFFIXES = ["bowl", "mug", "plate", "compartment", "handle"]


def normalize_token(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    text = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_.") or "none"


@dataclass(frozen=True)
class StructuredTarget:
    full: str
    core: str
    descriptor: str
    reference_relation: str
    reference_anchor: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def split_reference(target: str) -> tuple[str, str, str]:
    for relation, pattern in RELATION_PATTERNS:
        if pattern in target:
            left, right = target.split(pattern, 1)
            if left and right:
                return left, relation, right
    return target, "none", "none"


def split_core_descriptor(base: str) -> tuple[str, str]:
    for suffix in GENERIC_SUFFIXES:
        if base == suffix:
            return suffix, "none"
        token = f"_{suffix}"
        if base.endswith(token):
            descriptor = base[: -len(token)].strip("_")
            return suffix, descriptor or "none"
    return base, "none"


def parse_structured_target(target: object) -> StructuredTarget:
    normalized = normalize_token(target)
    base, reference_relation, reference_anchor = split_reference(normalized)
    core, descriptor = split_core_descriptor(base)
    return StructuredTarget(
        full=normalized,
        core=core,
        descriptor=descriptor,
        reference_relation=reference_relation,
        reference_anchor=reference_anchor,
    )
