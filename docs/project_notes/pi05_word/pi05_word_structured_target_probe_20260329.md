# pi05_word Structured-Target Probe Summary

## What changed

Instead of treating each descriptive target string as one flat class, the gold set target was automatically split into:

- `target_core`
- `target_descriptor`
- `target_reference_relation`
- `target_reference_anchor`

Example:

- `black_bowl_on_stove`
  - `target_core = bowl`
  - `target_descriptor = black`
  - `target_reference_relation = on`
  - `target_reference_anchor = stove`

## Structured gold-set preparation

- input gold records: `47`
- unique flat targets: `25`
- unique target cores: `19`
- unique descriptors: `5`
- unique reference relations: `5`
- unique reference anchors: `8`

This keeps descriptive information, but avoids forcing every composite target into a totally unrelated one-hot class.

## Structured probe setup

- backbone: `lerobot/pi05_libero_base`
- records: `47`
- split strategy: `core_aware`
- train size: `38`
- val size: `9`

## Final metrics

- `primitive_acc = 0.667`
- `relation_acc = 0.444`
- `target_core_acc = 0.444`
- `target_descriptor_acc = 0.778`
- `target_reference_relation_acc = 0.889`
- `target_reference_anchor_acc = 0.889`
- `target_exact = 0.333`
- `semantics_exact = 0.333`

Majority baselines:

- `primitive_majority_baseline = 0.222`
- `relation_majority_baseline = 0.333`
- `target_core_majority_baseline = 0.222`
- `target_descriptor_majority_baseline = 0.778`
- `target_reference_relation_majority_baseline = 0.889`
- `target_reference_anchor_majority_baseline = 0.889`

## Comparison to flat target probe

The key comparison is:

- flat target accuracy: `0.375`
- structured target-core accuracy: `0.444`

So the core object slot looks easier to recover than the original flat descriptive target class.

However, descriptor / reference slots are currently too imbalanced:

- most records have `descriptor = none`
- most records have `reference_relation = none`
- most records have `reference_anchor = none`

So those slot accuracies are currently close to majority baselines and should not be over-interpreted yet.

## Current takeaway

This structured version is still promising:

- it preserves descriptive target information
- it improves the recoverability of the target **core**
- it gives a more principled representation for future scaling

But to really test descriptive grounding, the next annotation round should intentionally add more:

- non-`none` descriptors
- non-`none` reference relations
- non-`none` reference anchors
