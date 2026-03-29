# pi05_spatial Checklist Report (2026-03-28)

## Goal

Based on the `pi05_spatial` checklist doc, determine whether `pi05_spatial` is:

1. directionally correct but currently too weak in implementation, or
2. largely irrelevant to the current benchmarks.

This report only keeps the highest-value findings.

## Completed Checks

- `1` task-level comparison on existing benchmarks
- `3` whether the geometry branch is actually used
- `4` whether the benefit is mainly early sample efficiency
- `3.2.4` anchor corruption / counterfactual sanity check

Not yet completed:

- failure-mode video taxonomy
- attention / local-sampling visualization
- stronger-constraint ablation
- geometry-sensitive subset relabeling

## 1. Benchmark Signal

### LIBERO-90 early / mid / later

| setting | pi05 | pi05_spatial | diff |
| --- | ---: | ---: | ---: |
| 500 step | 25.56% (23/90) | 26.67% (24/90) | +1.11 |
| 1000 step | 28.89% (26/90) | 33.33% (30/90) | +4.44 |
| 5000 step | 28.89% (26/90) | 32.22% (29/90) | +3.33 |
| 35000 step | 31.11% (28/90) | 30.00% (27/90) | -1.11 |

Interpretation:

- `pi05_spatial` does show a real early positive signal.
- That signal is not stable enough yet to become a later-stage advantage.
- Current evidence supports "helps early learning a bit" more than "raises final ceiling".

### LIBERO-Spatial

- official `pi05`: `97.0%` (`97/100`)
- `pi05_spatial@35k`: `97.0%` (`97/100`)

Only four tasks differ:

- base better: `task 1`, `task 4`
- spatial better: `task 5`, `task 7`

Interpretation:

- There is no strong or clean task-level pattern here yet.
- At least on this benchmark, the modification is not producing a clear systematic gain.

### LIBERO-90 task-level diff

At `5k`:

- spatial better: `9, 19, 33, 41, 59, 82`
- base better: `12, 47, 60`

At `35k`:

- spatial better: `28, 41, 44, 54, 79`
- base better: `9, 12, 13, 33, 38, 59`

Interpretation:

- There are some wins, but the winning tasks are not stable across training stage.
- The pattern is currently too mixed to argue that geometry-sensitive tasks are consistently benefiting.

## 2. Geometry Branch Is Only Weakly Coupled To Final Action

The most important implementation issue is in `modeling_pi05_spatial.py`:

- `decode_trajectory_to_actions(...)` currently returns `denoised_actions` directly
- the `trajectory_decoder` output is not used to form final action

So the intended path:

`pred_clean_trajectory -> trajectory_decoder -> pred_action`

is not actually active right now.

Direct consequence:

- the new geometry path can affect auxiliary representations and losses
- but it does not yet form a strong action constraint

## 3. Weight Check: Some New Modules Learn, But The Decoder Is Dead

Comparing `pi05_spatial` checkpoints at `500 step` and `35000 step`:

- `trajectory_decoder`: exactly unchanged
- `geometry_refinement`: changed
- `trajectory_tokenizer`: changed
- `spatial_feature_extractor`: changed

Observed gate values:

- `trajectory_tokenizer.residual_gate`: about `-0.00076 -> -0.00147`
- `geometry_refinement.token_gate`: about `0.00032 -> 0.00081`
- `geometry_refinement.delta_gate`: about `-0.00916 -> -0.05613`
- `trajectory_decoder.residual_gate`: `0.0 -> 0.0`

Interpretation:

- the geometry side-branch is not fully dead
- but its influence is still very weak
- the strongest intended coupling module, `trajectory_decoder`, is effectively not training

## 4. Counterfactual Anchor Test

Real-batch perturbation sanity check:

### 35k checkpoint

- `coord_abs_mean`: `3.34`
- `delta_abs_mean`: `2.01`
- `delta_to_coord_ratio`: `0.60`
- `shuffle_action_diff_ratio`: `0.0083`
- `const_action_diff_ratio`: `0.0080`

### 5k checkpoint

- `coord_abs_mean`: `5.68`
- `delta_abs_mean`: `1.71`
- `delta_to_coord_ratio`: `0.30`
- `shuffle_action_diff_ratio`: `0.0075`
- `const_action_diff_ratio`: `0.0074`

Interpretation:

- `delta_coord` is not collapsing to zero
- but scrambling anchors changes final action by less than `1%`
- this means the geometry branch is currently closer to a weak auxiliary hint than a behavior-defining signal

## 5. Checklist Verdict

Current evidence does **not** support "the direction is useless".

But it also does **not** support "the current implementation already works as intended".

The best-fitting conclusion is:

> the direction may have some value for early sample efficiency,  
> but the current implementation is too soft, and the key action-coupling path is not actually connected.

## 6. Continue Or Stop

### Reasonable to continue if

- the goal is to test whether stronger geometric constraints can improve behavior
- we first fix the action coupling and then rerun a small validation set

### Not reasonable to keep scaling the current version directly

Because three strong warning signs are already present:

1. later-stage task-level advantage is not stable
2. anchor corruption barely changes final action
3. `trajectory_decoder` is effectively unused

## 7. Highest-Value Next Step

Before any larger retraining, first make final action explicitly depend on the geometry path:

- use `pred_clean_trajectory`
- actually call `trajectory_decoder(...)`
- let `decode_trajectory_to_actions(...)` produce the final action from trajectory + state

Then rerun only:

1. anchor corruption test
2. `500 / 1000 / 5000` LIBERO-90 eval
3. one task-level comparison table

If anchor perturbation still has almost no effect after that, the line is much closer to "stop" than "continue".
