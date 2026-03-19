# Recent Experiment Summary (2026-03-17)

## Current Main Branches
- Original pi0 training run:
  - /mnt/sda/xzh/lerobot_outputs/pi0_pusht_20260313_201823
- Residual every-step alpha1 run:
  - /mnt/sda/xzh/lerobot_outputs/pi0_residual_everystep_alpha1_20260317_015201
- Residual every-step attn run (current active one):
  - /mnt/sda/xzh/lerobot_outputs/pi0_residual_every_step_attn_alpha15_20260317_173748

## Key Findings

### 1. Alpha sweep on residual every-step at checkpoint 50000 (50 episodes)
Baseline compare file:
- /home/XZH/projects/lerobot/outputs/eval/pi0_residual_everystep_alphaSweepSingle_ckpt050000_summary.json

Results:
- alpha=0.1: avg_sum_reward=65.6300, avg_max_reward=0.5123, pc_success=4.0
- alpha=0.5: avg_sum_reward=66.8103, avg_max_reward=0.5326, pc_success=0.0
- alpha=1.0: avg_sum_reward=62.3878, avg_max_reward=0.5510, pc_success=0.0
- alpha=1.5: avg_sum_reward=70.8306, avg_max_reward=0.5774, pc_success=6.0
- alpha=2.0: avg_sum_reward=62.2464, avg_max_reward=0.4606, pc_success=2.0

Conclusion:
- alpha=1.5 was the best overall setting among the tested alpha values.

### 2. Gate experiments at checkpoint 50000 (50 episodes)
Summary file:
- /home/XZH/projects/lerobot/outputs/eval/pi0_residual_everystep_gateAB_ckpt050000_summary.json

No-gate alpha=1.5 baseline:
- avg_sum_reward=70.8306, avg_max_reward=0.5774, pc_success=6.0

Experiment A gate:
- [1.0, 1.0, 0.9, 0.8, 0.7, 0.55, 0.4, 0.25, 0.15, 0.05]
- avg_sum_reward=66.1036, avg_max_reward=0.5221, pc_success=2.0

Experiment B gate:
- [1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.25, 0.12, 0.05, 0.0]
- avg_sum_reward=74.5838, avg_max_reward=0.5891, pc_success=0.0

Conclusion:
- Gate B improved reward, but did not improve success rate.
- No-gate alpha=1.5 remained the best setting for success rate.

### 3. Base postprocess experiments at checkpoint 50000 (50 episodes)
Summary file:
- /home/XZH/projects/lerobot/outputs/eval/pi0_residual_everystep_postprocessAB_ckpt050000_summary.json

Results:
- none: avg_sum_reward=70.8306, avg_max_reward=0.5774, pc_success=6.0
- shared: avg_sum_reward=63.1685, avg_max_reward=0.5138, pc_success=2.0
- segment-3: avg_sum_reward=60.0820, avg_max_reward=0.4986, pc_success=6.0

Conclusion:
- The original per-step base (none) remained the best overall.
- Shared and segment base postprocessing did not outperform the baseline.

### 4. Mean vs attn base pooling at checkpoint 10000 (50 episodes)
Results:
- attn:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_residual_every_step_attn_alpha15_ckpt010000_ep50/eval_info.json
  - avg_sum_reward=28.9727, avg_max_reward=0.3031, pc_success=2.0
- mean:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_residual_every_step_mean_ckpt010000_ep50/eval_info.json
  - avg_sum_reward=25.6072, avg_max_reward=0.2708, pc_success=2.0
- original pi0:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_ckpt010000_ep50/eval_info.json
  - avg_sum_reward=28.2352, avg_max_reward=0.3080, pc_success=0.0

Conclusion:
- Attention pooling clearly outperformed mean pooling at the same checkpoint.
- At 10000 steps, attn was already roughly on par with original pi0.

## Direct Comparisons Against Original pi0

### A. Original pi0 vs residual every-step alpha=1.5 at checkpoint 50000 (50 episodes)
- Original pi0:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_ckpt050000_20260317_ep50/eval_info.json
  - avg_sum_reward=57.0800
  - avg_max_reward=0.4363
  - pc_success=0.0
- Residual every-step alpha=1.5:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_residual_everystep_alphaSweepSingle_a1p5_ckpt050000_ep50/eval_info.json
  - avg_sum_reward=70.8306
  - avg_max_reward=0.5774
  - pc_success=6.0

Conclusion:
- At checkpoint 50000, residual every-step with alpha=1.5 was clearly better than original pi0.

### B. Original pi0 vs residual-attn vs residual-mean at checkpoint 10000 (50 episodes)
- Original pi0:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_ckpt010000_ep50/eval_info.json
  - avg_sum_reward=28.2352
  - avg_max_reward=0.3080
  - pc_success=0.0
- Residual attn:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_residual_every_step_attn_alpha15_ckpt010000_ep50/eval_info.json
  - avg_sum_reward=28.9727
  - avg_max_reward=0.3031
  - pc_success=2.0
- Residual mean:
  - /home/XZH/projects/lerobot/outputs/eval/pi0_residual_every_step_mean_ckpt010000_ep50/eval_info.json
  - avg_sum_reward=25.6072
  - avg_max_reward=0.2708
  - pc_success=2.0

Conclusion:
- At checkpoint 10000, residual-attn was already roughly on par with original pi0 and clearly better than residual-mean.
- The main earlier advantage over original pi0 was observed later, around checkpoint 50000 with alpha=1.5.

## Interpretation So Far
- Base is not strongest when it is largest; base quality matters more than raw magnitude.
- Attention pooling improved base usefulness even though v_base_norm became smaller.
- The base branch still behaves like a weak but useful bias term rather than the dominant component.
- The most promising current line is:
  - residual every-step
  - alpha around 1.5
  - attention pooling for base observation summary

## Current Active Training
- Resumed attn training log:
  - /home/XZH/projects/lerobot/logs/pi0_residual_every_step_attn_alpha15_resume_20260317_201242.log
