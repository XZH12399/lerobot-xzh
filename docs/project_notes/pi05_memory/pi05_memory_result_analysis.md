# PI05 Memory Result Analysis

Date: 2026-03-22
Project: /home/ct_24210860031/XZH/project/lerobot-xzh

## Background

Earlier comparisons suggested that `pi05_memory` was much worse than baseline `pi05`, and the first suspicion was that the explicit history retrieval branch was dragging performance down.

After re-checking the actual evaluation configs on A100, the main confounder turned out to be `n_action_steps`:

- baseline `pi05` checkpoint config: `chunk_size=50`, `n_action_steps=10`, `empty_cameras=1`
- `pi05_memory` checkpoint config: `chunk_size=50`, `n_action_steps=50`, `empty_cameras=0`

This means the earlier comparison was not fair, because `pi05_memory` was replanning much less frequently during rollout.

## Fairer Comparison So Far

The more comparable A100 evaluation is:

| Model | History | chunk_size | n_action_steps | empty_cameras | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| `pi05` | N/A | 50 | 10 | 1 | 90.0% |
| `pi05_memory` | on | 50 | 10 | 0 | 90.0% |
| `pi05_memory` | off | 50 | 10 | 0 | 86.0% |

Result files:

- baseline: `/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_overnight_final_eval_egl_20260321/eval_info.json`
- memory on (`n_action_steps=10`): `/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_memory_memfix2_hist_on_eval_egl_nact10_20260322/eval_info.json`
- memory off (`n_action_steps=10`): `/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_memory_memfix2_hist_off_eval_egl_nact10_20260322/eval_info.json`

## Interpretation

### 1. `n_action_steps=50` was a major confounder

This now looks like the strongest explanation for the previously bad `pi05_memory` result. In LIBERO, executing a full 50-step action queue before replanning is much more open-loop and accumulates error quickly.

### 2. `history on` is not inherently harmful

Under the aligned `n_action_steps=10` setting:

- `history on = 90.0%`
- `history off = 86.0%`

So the history branch is not simply corrupting inference. With more frequent replanning, it appears usable and may even help.

### 3. `history off` is still not equivalent to original `pi05`

Turning off `use_history_memory` in `pi05_memory` only disables the history augmentation path. It does not convert the model back into the original `pi05` architecture or weights.

So the fact that `memory off` is below baseline does **not** mean baseline is better than a true “originalized” `pi05_memory`; it only means the `pi05_memory` checkpoint without active retrieval is still not identical to original `pi05`.

### 4. The current result is encouraging, but still not perfectly controlled

This comparison is much better than before, but one mismatch remains:

- baseline `pi05`: `empty_cameras=1`
- `pi05_memory`: `empty_cameras=0`

So the next clean comparison should align `empty_cameras` as well.

## Empty Camera Aligned Rerun

The next rerun aligned `empty_cameras` as well:

| Model | History | chunk_size | n_action_steps | empty_cameras | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| `pi05` | N/A | 50 | 10 | 1 | 90.0% |
| `pi05_memory` | on | 50 | 10 | 1 | 94.0% |
| `pi05_memory` | off | 50 | 10 | 1 | 92.0% |

New result files:

- memory on (`empty_cameras=1`, `n_action_steps=10`): `/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_memory_memfix2_emptycam1_eval_egl_nact10_20260322_history_on/eval_info.json`
- memory off (`empty_cameras=1`, `n_action_steps=10`): `/home/ct_24210860031/XZH/project/lerobot-xzh/eval_outputs/pi05_family/pi05_memory_memfix2_emptycam1_eval_egl_nact10_20260322_history_off/eval_info.json`

Compared with the previous `empty_cameras=0` rerun:

- `history on`: `90.0% -> 94.0%`
- `history off`: `86.0% -> 92.0%`

This shows that `empty_cameras` was not a minor detail. It was another real confounder in the earlier comparisons.

## Updated Interpretation

### 5. `empty_cameras` alignment materially changes the result

Once `empty_cameras` is aligned to the baseline `pi05` setting, both `pi05_memory` variants improve. This means the earlier result was still underestimating the checkpoint quality.

### 6. Current evidence no longer supports “memory hurts” as the default story

Under the fairest comparison currently available on A100:

- baseline `pi05 = 90.0%`
- `pi05_memory history on = 94.0%`
- `pi05_memory history off = 92.0%`

So the current evidence is more consistent with:

- the `pi05_memory` checkpoint itself being healthy
- rollout configuration mismatches causing most of the earlier pessimistic read
- `history on` being at least competitive, and currently the best-performing setting among the tested variants

### 7. The gain from `history on` is positive but still modest

The new ranking is:

- `history on = 94.0%`
- `history off = 92.0%`
- baseline `pi05 = 90.0%`

This is encouraging, but the margin is not huge. Since each task only has 10 episodes, another rerun with more episodes would still be useful before making a very strong claim.

## Current Conclusion

A more up-to-date conclusion is:

- `pi05_memory` training is not failed
- the earlier poor result was strongly confounded by both `n_action_steps` mismatch and `empty_cameras` mismatch
- after aligning both of those settings, `pi05_memory` is competitive with, and currently slightly better than, the baseline `pi05` checkpoint on A100 LIBERO-10 eval
- `history on` is currently the best tested setting, but the margin over `history off` is still small enough that a larger-episode rerun would be valuable
