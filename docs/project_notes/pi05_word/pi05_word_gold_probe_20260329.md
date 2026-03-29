# pi05_word Gold-Set Probe Summary

## Inputs

- annotation source:
  - `outputs/pi05_word_annotation_round1/candidates.jsonl`
- cleaned gold set:
  - `outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence.jsonl`

## Gold-set cleaning

Cleaning rule:

- keep only records with complete annotation
- keep only `confidence = high`
- normalize labels to lowercase
- normalize `target` to snake_case

Cleaning result:

- input records: `50`
- cleaned high-confidence records: `47`
- dropped records: `3` (`confidence = medium`)

## Probe setup

- backbone: `lerobot/pi05_libero_base`
- tokenizer: `google/paligemma-3b-pt-224`
- dataset: `HuggingFaceVLA/libero`
- records used: `47`
- unique tasks: `26`
- unique episodes: `32`
- feature shape: `[47, 2048]`

Because the gold set is still small and many targets are singletons, validation used a **target-aware split**:

- singleton targets kept in train
- validation sampled only from targets with at least 2 examples

Split result:

- train size: `39`
- val size: `8`

## Final metrics

- `primitive_acc = 0.75`
- `target_acc = 0.375`
- `relation_acc = 0.5`
- `exact_match = 0.375`

Majority baselines:

- `primitive_majority_baseline = 0.125`
- `target_majority_baseline = 0.25`
- `relation_majority_baseline = 0.25`

## Interpretation

This gold-set probe still shows positive signal:

- `primitive` is clearly above baseline
- `relation` is above baseline
- `target` is only modestly above baseline, which is expected given the small sample size and the large number of distinct target labels

So the conclusion remains:

- the hidden representation appears to carry useful action-semantics information
- but the current gold set is still too small and too sparse in target classes for a strong target-grounding conclusion
