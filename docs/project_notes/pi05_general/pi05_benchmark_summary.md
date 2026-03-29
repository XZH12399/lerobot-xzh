# PI05 / PI05_Spatial Benchmark Summary

更新时间：2026-03-28

说明：
- 这里只保留关键实验，不追求完整流水账。
- 成功率统一取各自 `eval_info.json` 中的 `overall.pc_success`。
- `libero_90` 一栏里的 `lerobot/pi05_libero_finetuned` 只作为 sanity check，不应视为官方 `libero_90` baseline。

## 1. LIBERO-10

| 模型 | checkpoint / run | 评测设置 | 成功率 |
| --- | --- | --- | --- |
| `pi05` base | `pi05_libero10_overnight_20260320_overnight_base` @ `010000` | `libero_10`, 100 episodes | `90.0%` |
| `pi05_spatial` | `pi05_spatial_fixcfg_continue_20260327` @ `005000` | `libero_10`, 100 episodes | `94.0%` |

结论：
- `libero_10` 上，当前保留下来的正式结果里，`pi05_spatial@5k` 略高于本地 `pi05 base@10k`。
- 这个 benchmark 偏容易，很多 quick eval 很快接近饱和，不太适合放大结构改动差异。

关键文件：
- `eval_outputs/pi05_family/pi05_overnight_final_eval_egl_20260321/eval_info.json`
- `eval_outputs/pi05_family/pi05_spatial_fixcfg_continue_20260327_step005000_libero10_ep10_formal_rerun_20260327b/eval_info.json`

## 2. LIBERO-90

| 模型 | checkpoint / run | 评测设置 | 成功率 |
| --- | --- | --- | --- |
| `pi05` official sanity | `lerobot/pi05_libero_finetuned` | `libero_90`, 90 tasks x 1 ep | `27.8%` |
| `pi05` base | `pi05_libero_90_step005000_20260327_optimized_libero90_5k_quick` @ `005000` | `libero_90`, sharded, 90 tasks x 1 ep | `28.9%` |
| `pi05_spatial` | `pi05_spatial_libero_90_step005000_20260327_optimized_libero90_5k_quick` @ `005000` | `libero_90`, sharded, 90 tasks x 1 ep | `32.2%` |
| `pi05` base | `pi05_libero_90_step005000_20260327_optimized_libero90_5k_quick` @ `035000` | `libero_90`, sharded, 90 tasks x 1 ep | `31.1%` |
| `pi05_spatial` | `pi05_spatial_libero_90_step005000_20260327_optimized_libero90_5k_quick` @ `035000` | `libero_90`, sharded, 90 tasks x 1 ep | `30.0%` |

结论：
- `libero_90` 明显更难，成功率大致在 `28%` 到 `32%` 这个量级。
- `5k` 时 `pi05_spatial` 相比 `pi05 base` 有小幅优势；到 `35k` 时两者接近，且当前结果没有显示稳定领先。
- 这个 benchmark 更能拉开难度，但当前没有一个直接可对齐的官方 `pi05` `libero_90` checkpoint。

关键文件：
- `eval_outputs/pi05_family/pi05_official_libero90_ep1_sharded_20260328_sanity_a10/eval_info.json`
- `eval_outputs/pi05_family/pi05_libero_90_step005000_20260327_optimized_libero90_5k_quick_step005000_libero_90_ep1_sharded_20260328_001037_pi05/eval_info.json`
- `eval_outputs/pi05_family/pi05_spatial_libero_90_step005000_20260327_optimized_libero90_5k_quick_step005000_libero_90_ep1_sharded_20260328_001037_pi05_spatial/eval_info.json`
- `eval_outputs/pi05_family/pi05_libero_90_step035000_20260328_stoptrain_eval_fastseq_step035000_libero_90_ep1_sharded_20260328_35k_fastseq_pi05/eval_info.json`
- `eval_outputs/pi05_family/pi05_spatial_libero_90_step035000_20260328_stoptrain_eval_fastseq_step035000_libero_90_ep1_sharded_20260328_35k_fastseq_pi05_spatial/eval_info.json`

## 3. LIBERO-Spatial

| 模型 | checkpoint / run | 评测设置 | 成功率 |
| --- | --- | --- | --- |
| `pi05` official | `lerobot/pi05_libero_finetuned` | `libero_spatial`, 100 episodes | `97.0%` |
| `pi05_spatial` | `pi05_spatial_libero_90_step005000_20260327_optimized_libero90_5k_quick` @ `035000` | `libero_spatial`, 100 episodes | `97.0%` |

结论：
- 这是目前最干净的 head-to-head 对照，因为这里有可以直接使用的官方 `pi05` checkpoint。
- 当前结果上，两者总成功率打平，都是 `97.0%`。
- 逐 task 看有局部差异：`task 1/4` 官方 `pi05` 更好，`task 5/7` `pi05_spatial` 更好。

关键文件：
- `eval_outputs/pi05_family/pi05_official_libero_spatial_ep10_formal_20260328/eval_info.json`
- `eval_outputs/pi05_family/pi05_spatial_libero_spatial_step035000_ep10_formal_20260328/eval_info.json`
- 差异 task 视频对比：`eval_outputs/pi05_family/compare_reports/libero_spatial_pi05_spatial35k_vs_official_ep10_20260328/`

## 4. 当前建议

- 如果想看“是否接近官方 PI05”，优先看 `libero_spatial`。
- 如果想看“修改是否真的更抗难任务”，优先看 `libero_90`。
- `libero_10` 更适合做早期 smoke test，不适合单独作为最终结论。
