# pi05_word Probe Comparison Summary

This note consolidates the two manual-annotation validation runs completed on March 29, 2026:

- the flat gold-set probe
- the structured-target probe

The main question is whether `pi05_word` hidden features carry recoverable action semantics when the target label contains descriptive information, rather than only a coarse object name.

## 1. Gold set used by both experiments

Source:

- annotation candidates: `outputs/pi05_word_annotation_round1/candidates.jsonl`
- cleaned gold set: `outputs/pi05_word_annotation_round1/pi05_word_gold_high_confidence.jsonl`

Cleaning rule:

- keep only complete annotations
- keep only `confidence = high`
- normalize labels to lowercase
- normalize `target` to snake_case

Cleaning result:

| Item | Value |
| --- | ---: |
| input records | 50 |
| cleaned records | 47 |
| dropped records | 3 |
| unique tasks | 26 |
| unique episodes | 32 |

Gold-set label distribution:

- primitive: `approach=19`, `move_to=22`, `open=5`, `retreat=1`
- relation: `above=21`, `inside=10`, `on=10`, `none=5`, `near=1`
- flat targets: `25` unique labels

## 2. Experiment A: flat target probe

In this version, each full descriptive target string is treated as a single class, for example:

- `black_bowl_on_stove`
- `back_compartment_of_caddy`
- `yellow_and_white_mug`

Setup:

- backbone: `lerobot/pi05_libero_base`
- tokenizer: `google/paligemma-3b-pt-224`
- records: `47`
- feature shape: `[47, 2048]`
- split strategy: `target_aware`
- train size: `39`
- val size: `8`

Final metrics:

| Metric | Probe | Majority baseline |
| --- | ---: | ---: |
| primitive_acc | 0.750 | 0.125 |
| target_acc | 0.375 | 0.250 |
| relation_acc | 0.500 | 0.250 |
| exact_match | 0.375 | - |

Interpretation:

- `primitive` is clearly above baseline, so action-stage information is present.
- `relation` is also above baseline.
- `target` is only modestly above baseline because the label space is sparse: `25` target classes from only `47` records, with many singletons.

## 3. Experiment B: structured target probe

In this version, the original descriptive target is preserved but decomposed into slots:

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

Structured target summary:

| Item | Value |
| --- | ---: |
| full targets | 25 |
| target cores | 19 |
| descriptors | 5 |
| reference relations | 5 |
| reference anchors | 8 |

Important imbalance:

- descriptor counts: `none=34`, `black=6`, `drawer=5`, `back=1`, `yellow_and_white=1`
- reference relation counts: `none=40`, `next_to=3`, `on=2`, `from=1`, `of=1`
- reference anchor counts: `none=40`, all other anchors are singletons

Setup:

- backbone: `lerobot/pi05_libero_base`
- records: `47`
- split strategy: `core_aware`
- train size: `38`
- val size: `9`

Final metrics:

| Metric | Probe | Majority baseline |
| --- | ---: | ---: |
| primitive_acc | 0.667 | 0.222 |
| relation_acc | 0.444 | 0.333 |
| target_core_acc | 0.444 | 0.222 |
| target_descriptor_acc | 0.778 | 0.778 |
| target_reference_relation_acc | 0.889 | 0.889 |
| target_reference_anchor_acc | 0.889 | 0.889 |
| target_exact | 0.333 | - |
| semantics_exact | 0.333 | - |

Interpretation:

- `target_core` improves over the flat target result: `0.444` vs `0.375`.
- This suggests the hidden feature is better at recovering the object identity core than the full compositional string as a one-hot class.
- High descriptor / reference accuracies should not be over-interpreted because they are almost exactly the majority baseline.

## 4. Side-by-side comparison

| Question | Flat probe | Structured probe | Takeaway |
| --- | --- | --- | --- |
| Can the model recover primitive? | `0.750` vs baseline `0.125` | `0.667` vs baseline `0.222` | Yes, strong positive signal in both settings |
| Can the model recover relation? | `0.500` vs baseline `0.250` | `0.444` vs baseline `0.333` | Some signal, but still limited |
| Can the model recover target semantics? | full target `0.375` vs baseline `0.250` | core slot `0.444` vs baseline `0.222` | structured target is more learnable |
| Can the model recover descriptive target slots? | not available | descriptor / reference slots near baseline | current data is too imbalanced to conclude much |

## 5. Main conclusion

The key conclusion is not that descriptive targets are harmful. The more likely conclusion is:

1. `pi05_word` hidden features do contain usable action-semantics information.
2. Primitive information is easiest to recover.
3. Flat probing is too harsh for descriptive targets because each composite string becomes an isolated class.
4. A structured target probe is a better validation format for the user goal, because it preserves descriptive semantics while reducing unnecessary class fragmentation.

So if the research goal is "use descriptive targets because coarse object labels are not semantically sufficient", the structured direction is the more faithful one.

## 6. Limitations

The current study is still small and should be treated as a feasibility check, not a final claim.

Main limitations:

- only `47` high-confidence records
- validation sets are tiny: `8` samples for flat, `9` for structured
- many flat targets are singletons
- structured descriptor / reference slots are dominated by `none`
- several non-`none` structured labels appear only once

Because of this, the current experiments support:

- "there is probe-able signal"

but do not yet strongly support:

- "the model robustly grounds fine-grained descriptive target semantics"

## 7. Recommended next step

The best next validation step is not to remove description from the target. Instead:

1. keep the structured target format
2. collect more annotation cases with non-`none` descriptor values
3. collect more cases with non-`none` reference relation and anchor values
4. rebalance beyond frequent plain targets such as `basket`, `plate`, `bowl`, and `drawer_handle`
5. rerun the structured probe once each important slot has enough non-majority examples

If those additional samples are added, the next structured probe will give a much more trustworthy answer about whether `pi05_word` captures descriptive grounding.
