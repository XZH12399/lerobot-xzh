# pi05_word Round-1 Annotation Seed Export

## What was exported

Following `pi05_word_semantics_annotation_plan.md`, the next practical step was to prepare a small point-level annotation seed set:

- candidate snapshots for manual labeling
- three-frame context strips for quick confidence checking
- a JSONL manifest with empty annotation slots

## Output paths

Local workspace:

- `outputs/pi05_word_annotation_round1/candidates.jsonl`
- `outputs/pi05_word_annotation_round1/summary.json`

Server:

- `/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_word_annotation_round1/candidates.jsonl`
- `/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_word_annotation_round1/summary.json`
- `/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_word_annotation_round1/images/`
- `/home/ct_24210860031/XZH/project/lerobot-xzh/outputs/pi05_word_annotation_round1/context/`

## Export summary

- `num_candidates = 50`
- `semantic_family_counts = {open: 10, pick: 15, place: 25}`
- `unique_tasks = 27`
- `unique_episodes = 35`
- `wipe` is not available in the current LIBERO subset, so this round only covers `pick / place / open`

## Manifest format

Each JSONL record includes:

- `sample_id`
- `semantic_family`
- `phase_hint`
- `instruction`
- `episode_id`
- `frame_index`
- `image_path`
- `context_image_path`
- `annotation.primitive`
- `annotation.target`
- `annotation.relation`
- `annotation.confidence`
- `annotation.note`

The annotation fields are intentionally left `null` so this set can serve as a clean gold-set starting point.

## Recommended next manual step

Review `context_image_path` first, then annotate only high-confidence samples:

1. fill `primitive`
2. fill `target`
3. fill `relation`
4. set `confidence`
5. leave ambiguous samples empty or mark them for removal

Once enough high-confidence records are filled, we can rerun the hidden-state probe on the gold subset instead of heuristic labels.
