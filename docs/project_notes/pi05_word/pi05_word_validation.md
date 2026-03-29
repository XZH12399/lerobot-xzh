# Minimal Validation Plan: Predict Action Semantics from Current Image + Instruction on **LeRobot π0.5**

## Goal

Validate the first prerequisite of the proposed method on **LeRobot's π0.5 implementation**:

> Can the existing visual-language representations in LeRobot π0.5 predict the standardized action semantics we defined from the current image and task instruction?

This stage **does not** modify the flow-matching action head.  
It is a **pure perception-to-semantics** experiment:

\[
(o_t, g) \rightarrow y_t^{sem}
\]

where:

- \(o_t\): current image / observation
- \(g\): natural-language task instruction
- \(y_t^{sem}\): standardized action-semantics sequence

The purpose is to test whether the intermediate action-semantics representation is:

- **predictable** from existing π0.5 representations,
- **stable** enough to supervise,
- and **useful enough** to justify later integration into control.

---

## Important implementation assumption for LeRobot π0.5

This plan is written for the **LeRobot version of π0.5**, not for a full OpenPI reimplementation.

Therefore, the safest minimal route is:

- **do not assume** the repository already exposes a full free-form text-generation interface for the VLM,
- **do not assume** it is easy to add a new language-generation training objective end-to-end,
- **do assume** that it is easier to extract or reuse the hidden states produced from image + instruction conditioning,
- then attach a **minimal semantics prediction head** on top of those hidden states.

So the practical experiment is better phrased as:

> First verify whether LeRobot π0.5 hidden states already contain enough information to predict the proposed action semantics, by adding a lightweight semantics head while leaving the flow-matching action head untouched.

---

## Why this is the right first step

If we directly modify the action pathway, three hard questions get mixed together:

1. whether the action-semantics schema is learnable,
2. whether LeRobot π0.5 representations already encode enough information for it,
3. whether conditioning the action head on semantics improves control.

This plan isolates the first two questions and avoids entangling them with robot control.

---

## Key design decisions

### 1. Do **not** modify the FM head yet
Keep the existing LeRobot π0.5 action pathway unchanged.

### 2. Do **not** retrain the tokenizer first
For the minimal experiment:

- reuse the existing tokenizer if text decoding is attempted,
- or avoid tokenizer changes entirely by predicting structured labels from hidden states,
- only consider special tokens later if needed.

### 3. Prefer a **lightweight semantics head** over a full new text-generation stack
For LeRobot π0.5, the recommended first version is:

\[
h_t = f_{\pi0.5}(o_t, g)
\]
\[
\hat y_t^{sem} = \text{SemanticsHead}(h_t)
\]

where \(h_t\) are hidden states or pooled representations extracted from π0.5 after conditioning on image + instruction.

---

## Two implementation variants

## Variant A: Structured prediction head (**recommended first**)

Predict the action semantics as structured labels from hidden states.

Example factorization:

- step count prediction
- primitive sequence prediction
- slot prediction per step:
  - effector
  - primitive
  - target
  - relation
  - axis
  - speed
  - termination

This is the most realistic first experiment for LeRobot π0.5.

### Advantages
- avoids dependence on a free-form text decoder
- easier to train and evaluate
- easier to constrain outputs
- easier to debug

---

## Variant B: Restricted text decoding head

If LeRobot π0.5 already exposes a practical language-decoding path, or if adding a small LM-style decoding head is easy, then predict a restricted text DSL.

Example:

```text
ARM move_to target=bowl relation=above speed=medium END=pose_reached
GRIPPER open aperture=medium END=aperture_reached
ARM align target=bowl orientation=top_grasp END=pose_reached
ARM approach target=bowl axis=down speed=low END=contact_or_height_reached
GRIPPER close force=medium END=object_secured
ARM retreat direction=up speed=low END=lifted_clear
```

This is useful, but for LeRobot π0.5 it should be considered **secondary** to Variant A.

---

## Recommended experiment scope

### Inputs
- current RGB image (or the observation format already used in LeRobot π0.5)
- original natural-language instruction

### Output
A structured action-semantics prediction or restricted text-form action-semantics plan.

### Out of scope for this stage
- no FM head modification
- no control conditioning from semantics yet
- no robot deployment requirement
- no long-horizon planning

---

## Recommended action-semantics granularity

Use **Level 2 / medium granularity** only.

### Allowed effectors
- `ARM`
- `GRIPPER`

### Allowed primitives

#### ARM
- `move_to`
- `approach`
- `retreat`
- `align`
- `follow_path`
- `hold_pose`

#### GRIPPER
- `open`
- `close`
- `maintain`
- `release`

### Allowed fields
- `target`
- `relation`
- `axis`
- `direction`
- `speed`
- `aperture`
- `force`
- `orientation`
- `path`
- `END`

### Example field values
- `relation`: `above`, `near`, `contact`, `inside`, `on`
- `axis`: `down`, `up`, `left`, `right`, `forward`, `backward`
- `speed`: `low`, `medium`, `high`
- `force`: `light`, `medium`, `strong`
- `aperture`: `small`, `medium`, `large`
- `orientation`: `top_grasp`, `side_grasp`, `vertical`, `horizontal`, `handle_aligned`
- `END`: `pose_reached`, `aperture_reached`, `contact_or_height_reached`, `object_secured`, `object_released`, `lifted_clear`, `path_completed`, `timeout`

---

## Minimal target representation

## Option 1: Structured target (recommended)

Represent each step as a tuple:

```text
(effecter, primitive, slots..., termination)
```

Example:

```text
Step 1:
  effector = ARM
  primitive = move_to
  target = bowl
  relation = above
  speed = medium
  termination = pose_reached

Step 2:
  effector = GRIPPER
  primitive = open
  aperture = medium
  termination = aperture_reached
```

Then predict a short sequence of such steps.

### Advantages
- easiest to constrain
- no parsing errors
- easier to implement on top of LeRobot π0.5 hidden states

---

## Option 2: Restricted text target

If a text decoder is easy to expose, use:

```text
<EFFECTOR> <PRIMITIVE> key=value key=value ... END=<TERMINATION>
```

Example:

```text
ARM move_to target=bowl relation=above speed=medium END=pose_reached
GRIPPER open aperture=medium END=aperture_reached
ARM approach target=bowl axis=down speed=low END=contact_or_height_reached
GRIPPER close force=medium END=object_secured
ARM retreat direction=up speed=low END=lifted_clear
```

Use only if engineering effort is acceptable.

---

## Dataset plan

Since standard robot datasets typically do not contain this action-semantics annotation directly, build a **small seed dataset** first.

### Stage 1 dataset size
Start with **50 to 200 examples**.

### Task categories
Use only 4 short-horizon skill families at first:

1. `pick`
2. `place`
3. `open`
4. `wipe`

### Example instructions
- `pick up the bowl`
- `pick up the cup`
- `place the bowl on the table`
- `open the drawer`
- `wipe the table`

### Annotation strategy
Use **manual or semi-manual annotation** first.

For each example, create:
- image
- original instruction
- target action-semantics label

Example:

#### Input
- image: robot facing a bowl
- instruction: `pick up the bowl`

#### Label
```text
ARM move_to target=bowl relation=above speed=medium END=pose_reached
GRIPPER open aperture=medium END=aperture_reached
ARM align target=bowl orientation=top_grasp END=pose_reached
ARM approach target=bowl axis=down speed=low END=contact_or_height_reached
GRIPPER close force=medium END=object_secured
ARM retreat direction=up speed=low END=lifted_clear
```

---

## How to build labels efficiently

### Option A: Template-based labels (recommended first)
Map instruction types to predefined semantic templates.

#### Example
- `pick up X` -> grasp template
- `place X on Y` -> place template
- `open drawer` -> opening template
- `wipe surface` -> wiping template

Then fill object-specific parameters.

This is the most reliable first version.

### Option B: Trajectory-assisted refinement
If demonstrations are available, use trajectory statistics to refine the template:
- approach direction
- whether pregrasp is top or side
- whether retreat is upward
- whether the gripper opens before contact

This can be added later.

### Option C: LLM/VLM-assisted draft + human correction
Use a language model to draft candidate labels, then correct them manually.

Good for faster annotation once the schema stabilizes.

---

## Experimental stages

# Stage A: Hidden-state probing (**recommended first**)

## Objective
Test whether hidden states from LeRobot π0.5 already encode enough information to predict action semantics.

## Procedure
1. Run image + instruction through LeRobot π0.5.
2. Extract one or more hidden representations:
   - pooled VLM representation,
   - final hidden state of the instruction/image fusion,
   - or another stable intermediate representation already exposed by the codebase.
3. Train a lightweight semantics head on top.

## Example semantics head choices
- MLP for single-step slot prediction
- Transformer decoder for short semantic sequences
- autoregressive small decoder over discrete slot vocabulary

## Success criterion
If a lightweight head can recover semantics with good accuracy, then the representation is promising.

---

# Stage B: Zero-shot / few-shot text probing (optional)

## Objective
If a text-generation route is easy to expose, test whether the model can already output valid action-semantics text.

## Procedure
Prompt the model with:
- current image
- task instruction
- output schema
- optionally a few in-context demonstrations

## Evaluate
- syntax validity
- semantic plausibility
- target grounding
- consistency

This stage is optional because it may require more engineering than Stage A.

---

# Stage C: Supervised semantics prediction

## Objective
Train the semantics head to map image + instruction features to action-semantics targets.

## Training task
\[
(o_t, g) \rightarrow y_t^{sem}
\]

where the backbone is LeRobot π0.5 and only the semantics head is trained first.

## Recommended training setup
- backbone frozen at first
- semantics head trainable
- optionally unfreeze top layers later if needed

### Losses for structured prediction
Use a sum of cross-entropy losses:
- effector loss
- primitive loss
- slot losses
- termination loss
- optional sequence length loss

### Loss for text decoding
Standard token-level cross-entropy over the restricted DSL.

---

## Evaluation metrics

### 1. Syntax validity
For text format:
- legal effector
- legal primitive
- legal keys
- legal value domain
- legal `END`

For structured format:
- no invalid slot values
- no illegal primitive-slot combination

### 2. Step-level exact match
Useful but strict.

### 3. Slot accuracy
Per-field accuracy:
- correct target
- correct primitive
- correct relation
- correct axis
- correct termination

This is the most informative metric.

### 4. Semantic plausibility
Human evaluation:
- is the sequence sensible for the task?
- is the ordering valid?
- does the plan violate obvious manipulation constraints?

### 5. Visual grounding
Check whether referenced objects and relations match the image.

---

## Recommended baselines

### Baseline 1: Coarse Level-1 semantics
Use only coarse primitives:
- `move`
- `approach`
- `open`
- `close`
- `retreat`

Purpose:
- compare granularity tradeoff.

### Baseline 2: Level-2 semantics (main proposal)
The medium-grained schema proposed above.

### Baseline 3: Free-form natural-language decomposition
Ask for a natural-language decomposition instead of the structured schema.

Purpose:
- test whether the structure itself is the bottleneck.

### Optional Baseline 4: Level-3 semantics
Only if time allows.

Purpose:
- test whether the schema is too fine for LeRobot π0.5 representations.

---

## What to compare

The main question is:

> At what semantic granularity can LeRobot π0.5 representations reliably support image + instruction -> action semantics prediction?

So compare:
- Level 1
- Level 2
- optionally Level 3

across:
- syntax validity / structural validity
- slot accuracy
- human plausibility

---

## Minimal code changes for Codex

### Recommended first version
1. Identify where LeRobot π0.5 produces the fused representation for image + instruction.
2. Expose that representation.
3. Add a lightweight semantics head on top.
4. Add a dataset format for semantics labels.
5. Add training and evaluation scripts for the semantics task.

### Avoid for now
- do not change FM head
- do not redesign policy rollout code
- do not add a new tokenizer
- do not add semantics-conditioned action generation yet
- do not add robot deployment evaluation yet

---

## Suggested implementation order

### Step 1
Run LeRobot π0.5 forward pass and save the hidden states for image + instruction examples.

### Step 2
Train a small probe head for **single-step** semantics prediction first.

For example, predict only:
- primitive
- target
- relation

before predicting full multi-step semantics.

### Step 3
Expand to short semantic sequences:
- 3 to 7 steps per instruction

### Step 4
Compare Level 1 vs Level 2 semantics.

### Step 5
Only after good semantics prediction quality, consider integration into the control pathway.

---

## Suggested stage-gate criteria

Only continue to the next research step if Level-2 semantics reaches acceptable quality.

### Suggested thresholds
- structural validity > 90%
- slot accuracy reasonably high
- human plausibility clearly better than the coarse baseline

If these are not met, revise:
- the schema granularity,
- the label quality,
- the head design,
- the choice of hidden representation,
- or whether text decoding is necessary at all.

---

## What success means

This minimal experiment is successful if it shows:

1. LeRobot π0.5 hidden states contain enough information to predict the proposed action semantics,
2. Level-2 semantics is a workable granularity,
3. the semantics layer is feasible on top of the current LeRobot π0.5 codebase without touching the FM head.

This does **not** yet prove control improvement.  
It only proves that the proposed intermediate representation is feasible.

---

## Next step after success

If this stage succeeds, the next stage is:

\[
(o_t, g) \rightarrow y_t^{sem} \rightarrow a_t
\]

That is:
- inject the predicted action semantics into the action pathway,
- then test whether semantics-conditioned control improves execution.

But that is explicitly **after** this minimal validation stage.

---

## Deliverables for this stage

1. A small labeled dataset of image + instruction -> action semantics
2. A LeRobot π0.5 hidden-state extraction path
3. A lightweight semantics head
4. Hidden-state probing results
5. Level 1 vs Level 2 granularity comparison
6. A conclusion on whether the semantics layer is feasible on LeRobot π0.5
