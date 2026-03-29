# pi05_spatial Minimal Validation Verdict (2026-03-28)

## Scope

This note evaluates the current `pi05_spatial` "minimal validation" run against:

- `docs/project_notes/pi05_spatial/pi05_spatial 下一轮唯一值得做的最小验证方案.md`

The evaluated implementation is the **strong-coupling A-version**:

- final action is no longer taken directly from the FM denoised action branch
- final action is produced by `trajectory_decoder(pred_clean_trajectory, state)`

Relevant code:

- [src/lerobot/policies/pi05_spatial/modeling_pi05_spatial.py](D:\home\ct_24210860031\XZH\project\lerobot-xzh\src\lerobot\policies\pi05_spatial\modeling_pi05_spatial.py)

Key locations on A100:

- `embed_suffix(...)`: geometry alignment enters the denoising suffix tokens
- `decode_trajectory_to_actions(...)`: final action is fully replaced by `trajectory_decoder`
- `sample_actions(...)`: inference still performs FM denoising first, then geometry refinement, then decoder output

## What This Version Actually Tested

This version **did keep FM denoising**.

The actual inference chain is:

`noise -> FM denoise in action space -> denoised action-like sequence -> geometry refinement -> 3D trajectory -> trajectory_decoder -> final action`

So this is **not** a pure "latent-only geometry alignment while keeping FM final action unchanged" test.

It is instead a **behavior-coupling stress test**:

- does geometry now affect final action?
- can the decoder-taken-over policy still behave correctly?

## Evidence Collected

### 1. Geometry now affects final action

Old `pi05_spatial` anchor corruption was previously below `1%`.

Current strong-coupling run:

- `step 500`
  - `shuffle_action_diff_ratio = 0.0474`
  - `const_action_diff_ratio = 0.0267`
- `step 1000`
  - `shuffle_action_diff_ratio = 0.0264`
  - `const_action_diff_ratio = 0.0210`

Interpretation:

- the geometry path is no longer an almost-ignored hint
- it now measurably changes final action

### 2. Decoder is active, not dead

Training logs show:

- `trajectory_decoder_gate` stayed around `1.000 -> 1.014`
- `decoder_action_abs_mean` stayed nonzero
- the decoder is clearly in the forward path

Interpretation:

- the "dead decoder" problem from the earlier soft-coupling version is fixed

### 3. Behavior collapsed at `500 step`

Formal `LIBERO-90` eval for the current coupled version:

- `step 500`: `0.0%` (`0/90`)

Reference baselines:

- base `pi05@500`: `25.56%` (`23/90`)
- old `pi05_spatial@500`: `26.67%` (`24/90`)

Task-level comparison against base at `500`:

- coupled better than base: none
- base better than coupled: `23` tasks

Interpretation:

- this version is not just slightly worse
- it is behaviorally broken at early stage

### 4. `1000 step` partial eval shows the same failure pattern

Artifacts on A100 before the run was stopped:

- `anchor_step001000.json` was completed
- `step001000` eval directory contains partial videos only, no final `eval_info.json`

Manual video review for `step 500` and partial `step 1000` showed the same pattern:

- arm moves slowly in one direction
- little task awareness
- gripper shows almost no useful response

Interpretation:

- the failure is not limited to one bad rollout
- it persists into the next checkpoint

## Mapping To The Minimal-Validation Criteria

### Passed

- `Condition 1`: anchor corruption now clearly affects final action
- `Condition 2`: `trajectory_decoder` is clearly active

### Failed

- `Condition 3`: early positive eval signal did not hold
  - current coupled `500` result is `0/90`
- `Condition 4`: no stable useful task-level advantage pattern emerged

## Verdict

For the **current strong-coupling A-version**, the conclusion is:

> mechanism coupling is now real,  
> but this implementation is behaviorally unusable.

This means:

- the previous diagnosis was correct: weak coupling was a real problem
- but the current fix is too aggressive for the intended research goal

## What Should Be Kept

- keep the finding that geometry must influence the behavior path more directly than before
- keep the geometry-aware processing inside the FM suffix / denoising path
- keep the anchor-corruption sanity check as a standard diagnostic

## What Should Be Rejected

- do not continue scaling this exact "decoder fully takes over final action" version
- do not use this run as evidence against the broader geometry-alignment idea

## Most Likely Correct Next Direction

The user-intended direction is:

- align FM latent/action tokens with image-space geometry **during denoising**
- but keep final action primarily on the FM branch
- do not force a post-hoc trajectory decoder to fully replace the final action

In short:

- **keep geometry inside FM**
- **do not let the post-trajectory decoder fully own final control**

## Bottom Line

This run successfully answered one narrow question:

- yes, once geometry truly controls final action, anchor corruption is no longer negligible

But it also answered the more practical question:

- this specific takeover-style implementation should be stopped

So the line that should continue is:

- **latent-space / denoising-stage alignment**

The line that should stop is:

- **post-hoc trajectory-decoder full takeover**
