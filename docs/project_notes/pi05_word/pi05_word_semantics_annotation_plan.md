# Action-Semantics Annotation Plan for LeRobot π0.5

## Goal

Build a **small, high-confidence gold annotation set** for short-horizon action semantics, so we can validate whether LeRobot π0.5 can predict a standardized semantics layer from **current image + instruction** before touching the flow-matching action head.

This stage is for **data and tooling preparation**, not control integration.

---

## Why start with manual annotation

At this stage, the main uncertainty is not model capacity. It is **label correctness**:

- What exactly counts as `approach` vs `move_to`?
- At a given frame, is the target the **object being grasped** or the **destination being approached**?
- Which frames are “clean” enough to support a single-step semantics label?
- If action semantics is meant to guide control later, how should a **semantic unit align with a short action trajectory segment**?

A small, carefully curated **gold set** is much more useful than a large noisy set.

---

## Recommended scale

Start with:

- **50 to 100 labeled samples** for the first round
- expand to **200 samples** only after the schema is stable

Use only **high-confidence frames**.

---

## Task scope for round 1

Use only 4 short-horizon skill families:

1. `pick`
2. `place`
3. `open`
4. `wipe`

This keeps the annotation rules simple and makes it easier to debug the semantics schema.

---

## Two-layer annotation strategy

To keep the project manageable while still staying aligned with the long-term goal, use **two layers of annotation**.

### Layer A: Point-level annotation
This is the first stage and is used for:
- hidden-state probing
- semantics feasibility testing
- checking whether the label schema is stable

Each sample corresponds to **one selected frame** and one single-step semantics label.

### Layer B: Segment-level annotation
This is the second stage and is used for:
- aligning one action semantics unit with a **short action trajectory**
- training semantics-to-action-chunk models
- building a true control interface

Each sample corresponds to:
- a trajectory segment start
- a trajectory segment end
- one semantics label for the whole segment

### Recommendation
Start with **Layer A only**.  
Once the semantics schema is stable, add **Layer B** on a smaller subset.

---

## Label schema for round 1

Use **single-step action semantics** only.

### Required fields
- `primitive`
- `target`
- `relation`

### Optional fields for round 2
- `speed`
- `orientation`
- `termination`

### Example labels

```text
primitive = approach
target = bowl
relation = above
```

```text
primitive = move_to
target = basket
relation = inside
```

```text
primitive = move_to
target = plate
relation = on
```

---

## Annotation principle

Only label samples where the action semantics is **visually and temporally unambiguous**.

### Good examples
- gripper clearly approaching an object from above before grasp
- object already grasped and end-effector clearly moving toward destination
- handle clearly visible and robot clearly moving toward it
- wiping tool already in contact with surface and sweeping over it

### Bad examples
- ambiguous transition frames between grasp and transport
- frames where the object is heavily occluded
- frames where target identity is unclear
- frames where multiple semantics are equally plausible

If a sample feels ambiguous, skip it.

---

## Step-by-step workflow

## Step 1: Define annotation rules

Create a short annotation guide before labeling begins.

It should define:

### `primitive`
Recommended first-round vocabulary:
- `approach`
- `move_to`

Optional later:
- `align`
- `retreat`
- `follow_path`

### `target`
The main object or destination currently relevant to the step.

Examples:
- `bowl`
- `basket`
- `plate`
- `drawer_handle`
- `table_surface`

### `relation`
Recommended first-round vocabulary:
- `above`
- `inside`
- `on`
- `near`
- `contact`

### Rule examples
- If the end-effector is moving toward an object to grasp it from above, use:
  - `primitive=approach`
  - `target=<object>`
  - `relation=above`

- If the object is already grasped and the robot is transporting it toward a container or placement surface, use:
  - `primitive=move_to`
  - `target=<destination>`
  - `relation=inside` or `on`

---

## Step 2: Select candidate episodes

From the LeRobot / LIBERO dataset, select only episodes from the four task families above.

For each episode, identify a few candidate frames:
- one near the **grasp phase**
- one near the **place/open/wipe phase**

Do not densely annotate all frames.

---

## Step 3: Extract annotation snapshots

For each candidate frame, prepare a compact record containing:
- image path or image tensor reference
- instruction text
- episode id
- frame index / timestamp
- optional nearby frames for context

This makes annotation easier and avoids constantly re-opening raw trajectories.

---

## Step 4: Manually annotate high-confidence samples

For each sample:
1. inspect image
2. read instruction
3. assign `primitive`, `target`, `relation`
4. add a confidence flag:
   - `high`
   - `medium`
   - `low`

Only keep **high-confidence** examples in the first gold set.

Suggested JSON record:

```json
{
  "episode_id": 826,
  "frame_index": 123,
  "instruction": "pick up the chocolate pudding and place it in the basket",
  "primitive": "approach",
  "target": "chocolate_pudding",
  "relation": "above",
  "confidence": "high"
}
```

---

## Step 5: Run label consistency check

Before training anything:

- review all labels once more
- check for inconsistent use of `approach` vs `move_to`
- check whether similar visual states got similar labels
- remove uncertain examples

If possible, have a second pass by another annotator or by yourself after a time gap.

---

## Step 6: Train the first probe or generation model

Once the gold set is ready, use it for one of two validation routes:

### Route A: Probe
Freeze LeRobot π0.5 backbone and train a lightweight classifier on top of hidden states.

This tells you whether existing representations already encode the semantics labels.

### Route B: Direct generation
Constrain the VLM to output restricted semantics text directly.

This tells you whether the model can “say” the semantics, not just encode it internally.

---

# Part II: Segment-level annotation plan

## Why segment-level annotation is needed

Point-level labels are enough for:
- probe experiments
- hidden-state analysis
- basic semantics feasibility checks

But they are **not enough** if the semantics layer is eventually meant to guide control.

Many action semantics correspond not to a single instant, but to a **short trajectory chunk**:

- `approach bowl from above`
- `move_to plate`
- `descend toward object`
- `wipe along surface`

So for control-aligned training later, one semantics unit should correspond to **one short action segment**.

### Core principle

\[
	ext{one action semantic} \leftrightarrow 	ext{one short action trajectory segment}
\]

---

## When to start segment annotation

Do **not** begin with segment annotation at full scale.

Recommended order:

1. first stabilize the semantics schema with point-level labels
2. then annotate a **small subset** of episodes with segment boundaries

This avoids committing to unstable semantics definitions too early.

---

## What a segment annotation should contain

Each segment should include:
- `episode_id`
- `start_frame`
- `end_frame`
- `primitive`
- `target`
- `relation`
- optional `confidence`
- optional `note`

Suggested JSON format:

```json
{
  "episode_id": 826,
  "start_frame": 120,
  "end_frame": 138,
  "instruction": "pick up the chocolate pudding and place it in the basket",
  "primitive": "approach",
  "target": "chocolate_pudding",
  "relation": "above",
  "confidence": "high"
}
```

---

## How to choose segment boundaries

Do **not** cut trajectories at fixed length only.

Use **semantic boundaries**, mainly when one of the following changes:

### 1. End-effector motion mode changes
Examples:
- horizontal move -> vertical descent
- descent -> lift
- pull -> retreat

### 2. Gripper state changes
Examples:
- open -> close
- close -> maintain

### 3. Object relation changes
Examples:
- `above bowl`
- `contact bowl`
- `on plate`
- `inside basket`

### 4. Interaction/contact state changes
Examples:
- before contact
- contact begins
- object secured
- object released

### Practical rule
A new segment starts when the **local function of the motion** changes.

---

## Recommended segment granularity

A segment should be:

- **long enough** to represent a complete local control process
- **short enough** to contain only one local function

### Too short
- one or two timesteps
- no visible motion pattern

### Too long
- approach + grasp + lift all merged into one segment

### Reasonable first guess
Use short chunks that roughly match one local phase, such as:
- pregrasp move
- descent / approach
- grasp
- lift
- transport to destination
- placement approach

If needed, later estimate segment duration statistics from demonstrations.

---

## Example: segmenting a grasp trajectory

Task:
`pick up the bowl`

Possible segment decomposition:

### Segment 1
- `primitive = move_to`
- `target = bowl`
- `relation = above`

Meaning:
robot moves to a pregrasp pose above the bowl

### Segment 2
- `primitive = approach`
- `target = bowl`
- `relation = above`

Meaning:
robot descends toward the bowl

### Segment 3
- `primitive = move_to` or later `grasp`
- `target = bowl`
- `relation = contact`

Meaning:
robot closes on the object / reaches grasp contact

### Segment 4
- `primitive = retreat`
- `target = bowl`
- `relation = above`

Meaning:
robot lifts the object away from the table

---

## Recommendation for round 2

Once point labels are stable, annotate **10 to 20 episodes** with segment boundaries.

This is enough to test:
- whether one semantics unit aligns with one short action chunk
- whether segment-level supervision is feasible
- whether the semantics layer can later be connected to action generation

Do not scale segment annotation before these questions are answered.

---

## Recommended annotation tool

Yes — a small annotation tool will make this much easier.

### Option 1: Use an existing annotation platform
Recommended if you want to label images comfortably and export JSON.

#### CVAT
Pros:
- mature UI
- image annotation support
- import/export workflows
- can scale later

Cons:
- heavier setup than needed for a tiny first pass

#### Label Studio
Pros:
- easier to customize for mixed image + text labeling
- convenient for custom form-style interfaces
- good fit for labeling `primitive`, `target`, `relation`

Cons:
- still more setup than a minimal custom tool

### Option 2: Build a tiny local annotation tool (**recommended first**)
For the first 50–100 samples, the best option is probably a **simple local tool**:

- left panel: image
- top text: instruction
- dropdowns:
  - primitive
  - target
  - relation
  - confidence
- buttons:
  - save
  - skip
  - previous / next

Export directly to JSONL.

Why this is best now:
- much faster to implement
- no heavy deployment
- exactly matches your schema
- ideal for a high-confidence gold set

---

## Recommended tool decision

### For the first round
Use a **small custom local annotation UI**.

### Later, if scale grows
Move to **Label Studio** first, because it is flexible for custom image + text + categorical labeling workflows.

Use **CVAT** if you later need more video-style or computer-vision-heavy annotation workflows.

---

## Extra UI requirements for segment annotation

The annotation tool should later support a **segment mode** in addition to point mode.

### Segment mode should allow:
- browse an episode trajectory
- display current frame and neighboring frames
- mark `start_frame`
- mark `end_frame`
- assign `primitive`
- assign `target`
- assign `relation`
- assign `confidence`
- save segment annotation

### Recommended display for segment mode
- center: current frame
- bottom: frame scrubber / slider
- left: instruction
- right: annotation form
- optional mini-strip of nearby frames

This will make trajectory segmentation much easier than labeling from isolated images only.

---

## Suggested file outputs

Codex should prepare:

1. `annotation_guidelines.md`
   - definitions of primitive / target / relation
   - examples and edge cases

2. `annotation_samples.jsonl`
   - candidate point-level samples for labeling

3. `annotation_segments_candidates.jsonl`
   - candidate episodes and frame ranges for segment labeling

4. `annotation_tool/`
   - minimal local UI for manual labeling
   - support both point mode and segment mode

5. `gold_annotations.jsonl`
   - final high-confidence point labels

6. `gold_segment_annotations.jsonl`
   - final high-confidence segment labels

7. `label_audit_report.md`
   - summary of counts, skipped samples, low-confidence cases

---

## Minimal UI specification for Codex

The annotation tool should support:

### Point mode input
- a JSONL file with:
  - sample id
  - image path
  - instruction
  - episode id
  - frame index

### Point mode display
- current image
- instruction text
- sample metadata

### Point mode controls
- dropdown: `primitive`
- dropdown: `target`
- dropdown: `relation`
- dropdown: `confidence`
- text field: optional note
- buttons: `Save`, `Skip`, `Prev`, `Next`

### Segment mode input
- episode id
- ordered frame references
- instruction

### Segment mode display
- current frame
- timeline slider
- neighboring frames

### Segment mode controls
- mark `start`
- mark `end`
- dropdown: `primitive`
- dropdown: `target`
- dropdown: `relation`
- dropdown: `confidence`
- note field
- buttons: `Save Segment`, `Skip`, `Prev Episode`, `Next Episode`

### Output
Append one JSON object per labeled item to a JSONL file.

---

## Quality control rules

Codex should enforce these checks:
- no missing required fields
- `primitive`, `target`, `relation` must be in allowed vocab
- no duplicate sample ids in final point-label output
- no duplicate `(episode_id, start_frame, end_frame)` triples in segment output
- keep confidence as explicit field
- maintain a separate skipped-samples list

---

## Suggested milestone plan

### Milestone 1
Write annotation guidelines.

### Milestone 2
Prepare 50 candidate point-level samples from LeRobot / LIBERO.

### Milestone 3
Build local annotation UI with point mode.

### Milestone 4
Annotate 50 high-confidence point labels.

### Milestone 5
Audit label consistency.

### Milestone 6
Use the point-level gold set for probe or direct-generation validation.

### Milestone 7
Prepare 10–20 candidate episodes for segment annotation.

### Milestone 8
Extend UI with segment mode.

### Milestone 9
Annotate a small segment-level gold set.

### Milestone 10
Use the segment-level set for semantics-to-action-chunk experiments.

---

## Final recommendation

Yes — your next step should still be:

1. define a strict single-step semantics labeling guide,
2. build a small custom annotation tool,
3. label a **small high-confidence point-level gold set**,
4. then use that set for probe or direct-generation validation.

After that, move to:

5. annotate a **small segment-level set**,
6. align one semantics unit with one short action trajectory,
7. use that for control-oriented experiments.

Do **not** start by labeling the whole dataset.
Do **not** start by automating everything.
Start with a small, clean set that makes the semantics definition stable, then extend it from **point labels** to **segment labels**.
