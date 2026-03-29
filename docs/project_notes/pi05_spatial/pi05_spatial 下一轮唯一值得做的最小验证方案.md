# pi05_spatial 下一轮唯一值得做的最小验证方案

> 目标：  
> 不再继续扩大当前版本实验，而是先回答一个更根本的问题：
>
> **如果把 geometry path 真正接到 final action 上，模型会不会开始依赖它，并带来可验证的行为变化？**
>
> 这轮实验的性质不是“继续打榜”，而是一次 **去留判定实验**。  
> 如果这轮仍然失败，就应当非常接近止损。

---

# 0. 本轮实验的总原则

## 0.1 只做“让 geometry path 变成必要路径”的最小修改

本轮不做以下事情：

- 不继续在当前版本上直接拉长训练
- 不增加新 benchmark
- 不再加新的复杂模块
- 不做新的大规模结构扩张
- 不尝试同时修改多个因素

本轮只做一件核心事情：

**让 final action 显式依赖 geometry path 的输出。**

---

## 0.2 本轮不是追求最终最优性能，而是追求“机制是否成立”

这一轮最想看到的，不是最终一定超过 baseline 多少点，而是下面这些机制信号：

1. anchor 被打乱后，final action 明显变化  
2. `trajectory_decoder` 开始真正训练  
3. 几何路径对 early eval 有更稳定影响  
4. 相比当前版本，几何路径从“弱辅助 hint”变成“行为相关信号”

只要这些信号出现，这条线就还值得继续。  
如果这些信号依然不出现，就应该高度倾向停止。

---

# 1. 本轮要修改的核心问题

根据当前排查，现版本的关键问题是：

- 理论上你希望有一条路径：

    pred_clean_trajectory -> trajectory_decoder -> pred_action

- 但实际上目前 `decode_trajectory_to_actions(...)` 直接返回 `denoised_actions`
- 所以 `trajectory_decoder` 输出没有真正参与 final action 形成  
- 结果就是 geometry path 只影响 auxiliary representation / loss，而不是 final behavior

所以本轮只围绕这个问题做修复。

---

# 2. 本轮修改目标

## 2.1 一级目标

让 final action 显式由以下信息共同决定：

- `pred_clean_trajectory`
- `robot_state`
- （可选）`denoised_actions` 作为残差或辅助项

即从当前的：

`final_action ≈ denoised_actions`

改成：

`final_action = action_from_geometry_path(...)`

或至少：

`final_action = geometry_action + residual_from_denoised_action`

---

## 2.2 二级目标

让 `trajectory_decoder` 成为真正会训练、会影响输出的模块。  
当前最不可以接受的情况是：

- `trajectory_decoder` 参数不变
- `trajectory_decoder.residual_gate` 长期为 0
- `trajectory_decoder` 输出对 final action 没有贡献

本轮后应至少看到：

- `trajectory_decoder` 权重发生变化
- 它的 gate / residual 不再恒为 0
- 它对 action 输出产生可测影响

---

# 3. 推荐的最小实现方式

下面给出三个强度版本。  
建议优先做 **版本 A**，必要时再做 **版本 B**。  
**不建议一开始就上版本 C。**

---

## 3.1 版本 A：直接替换 final action（最干净、最有判别力）

### 做法

让：

`pred_action = trajectory_decoder(pred_clean_trajectory, robot_state)`

并让 `decode_trajectory_to_actions(...)` 返回这个 `pred_action`，  
而不是直接返回 `denoised_actions`。

### 优点

- 判别力最强
- 最容易看出 geometry path 是否真的有用
- 不会再被“主干绕开”

### 风险

- 如果 decoder 还不稳定，可能短期性能下降
- 但这轮实验目标本来就不是追求最高分，而是验证机制

### 推荐程度

**最高。**

因为你现在最缺的不是一个稳妥优化版本，而是一个能回答“geometry path 有没有行为控制力”的版本。

---

## 3.2 版本 B：geometry action 为主，denoised action 为残差（更稳妥）

### 做法

令：

`geometry_action = trajectory_decoder(pred_clean_trajectory, robot_state)`

`pred_action = geometry_action + alpha * residual_action_from_denoised`

其中：

- `alpha` 初始设小一点，比如 `0.1 ~ 0.3`
- residual 分支来自原来的 `denoised_actions`

### 优点

- 比版本 A 更稳
- 让 geometry path 成为主路径，同时保留部分原始 action 能力

### 风险

- 如果 residual 太强，主干仍可能绕开 geometry path
- 所以 `alpha` 不宜过大

### 推荐程度

**次高。**

如果你担心版本 A 太激进，可以先用 B。

---

## 3.3 版本 C：门控融合（当前阶段不建议优先）

### 做法

`gate = sigmoid(g(...))`

`pred_action = gate * geometry_action + (1 - gate) * denoised_action`

### 问题

- 很容易再次退化成“网络学会只用 denoised 分支”
- 你现在最怕的就是这个

### 推荐程度

**低。**

除非 A/B 验证出 geometry path 有价值，再回来做更精细的融合。

---

# 4. 本轮具体修改清单

---

## 4.1 修改点一：final action 生成逻辑

### 当前状态

类似于：

`def decode_trajectory_to_actions(...):`
  
`    return denoised_actions`

### 本轮目标

改成类似：

`def decode_trajectory_to_actions(pred_clean_trajectory, robot_state, denoised_actions=None):`
  
`    geometry_action = trajectory_decoder(pred_clean_trajectory, robot_state)`
  
`    # 版本 A`
  
`    return geometry_action`
  
`    # 或版本 B`
  
`    # return geometry_action + alpha * residual_from_denoised(denoised_actions)`

### 成功标准

- `trajectory_decoder` 确实参与 forward
- final action tensor 对 `pred_clean_trajectory` 有梯度依赖
- 去掉 `pred_clean_trajectory` 时，final action 会变

---

## 4.2 修改点二：确保 `trajectory_decoder` 参数真的进入优化

### 要检查的点

- optimizer param groups 里有没有 `trajectory_decoder`
- 是否被冻结
- 是否被 `requires_grad=False`
- 是否被错误地 `detach`
- forward 时输入是否非零、非恒定
- loss 是否能回传到 decoder

### 训练前必须确认

至少打印一次：

- `trajectory_decoder` 参数总数
- 可训练参数数
- 梯度范数是否非零

### 成功标准

训练几十到几百 step 后：

- `trajectory_decoder` 参数发生变化
- 对应 gate 不再永远是 0

---

## 4.3 修改点三：检查 `pred_clean_trajectory` 本身是否有信息量

因为即使接上了 decoder，如果 `pred_clean_trajectory` 本身无信息，也没用。

### 必做检查

在训练前后抽样打印：

- `pred_clean_trajectory` 均值 / 方差
- 不同 batch 间是否变化
- 不同 task 间是否变化
- 与 GT trajectory / anchor 是否有粗相关性

### 成功标准

- `pred_clean_trajectory` 不是常量
- 它随输入变化
- 它对 task / scene 有分辨性

---

## 4.4 修改点四：必要时降低几何辅助损失权重，避免“学辅助目标、不学动作”

当前已有信号说明 spatial 版本的 total loss 很大，而 action_loss 没有同步反映全部问题。  
这很可能意味着 auxiliary losses 太重，容易出现：

- 模型努力优化几何分支内部指标
- 但对 final control 没有帮助

### 建议

如果你接上 decoder 后发现训练不稳，可先尝试：

- 保持 action loss 为主
- 适当减小 `coord / proj / smoothness` 的权重
- 让 geometry path 优先学会“为 action 服务”

### 目标

不要让 geometry path 继续沦为“自洽但无用”的支路。

---

# 5. 本轮训练与评测的最小配置

本轮不要跑大而全。  
只跑一个足够判断去留的最小闭环。

---

## 5.1 训练 checkpoint

建议保存并评测：

- `500 step`
- `1000 step`
- `5000 step`

这三个点就够了。  
理由：

- `500 / 1000`：看 early sample efficiency 是否仍存在、是否更强
- `5000`：看中期信号是否比当前版本更稳定

**不建议一开始就跑到 `35000`。**

---

## 5.2 benchmark

只跑：

- `LIBERO-90`

理由：

- 它比 `LIBERO-10` 更有区分度
- 比 `LIBERO-Spatial` 更不容易天花板饱和
- 你前面的 early positive signal 也主要出现在这里

---

## 5.3 评测配置

仍然先用：

- `90 task × 1 episode`

因为这轮是机制验证，不是正式论文结果。  
如果出现强正信号，再补更稳的重复评测。

---

# 6. 本轮必须做的三个诊断实验

这三项必须和训练一起做。  
否则即使分数有变化，也很难判断原因。

---

## 6.1 诊断 A：anchor corruption test（最高优先级）

### 做法

对训练后的 checkpoint，在推理时分别做：

1. 正常 anchor
2. shuffle anchor
3. constant anchor

比较：

- final action difference ratio
- eval success drop

### 当前旧版本的问题

当前版本里，这个 ratio 只有 `<1%`，说明几何路径几乎不影响 final action。

### 本轮目标

修正后，这个值应显著上升。

### 推荐判断线

不是要求一个绝对数字，但至少应满足：

- 明显高于当前 `<1%`
- 并且 eval success 对 anchor corruption 更敏感

### 止损条件

如果修正 coupling 后，anchor 打乱仍几乎不影响 final action，  
那这条线非常接近该停。

---

## 6.2 诊断 B：decoder weight / gate tracking

### 做法

比较：

- `step 0`
- `500 step`
- `1000 step`
- `5000 step`

跟踪：

- `trajectory_decoder` 参数变化量
- decoder 对应 gate 数值
- decoder 输出范数
- decoder 输出对 final action 的贡献比例

### 本轮目标

必须看到：

- decoder 参数开始更新
- gate 不再恒等 0
- decoder 输出不是近零噪声

### 止损条件

如果 `trajectory_decoder` 接上 final action 后仍然基本不训练，那说明实现或变量设计更深层有问题。

---

## 6.3 诊断 C：task-level diff table

### 做法

在 `1000 step` 和 `5000 step`，至少做一份：

- spatial better tasks
- base better tasks

### 目的

这次不要求一开始就证明“几何敏感任务稳定获益”，  
但至少要看 pattern 是否比当前版本更稳定。

### 本轮希望看到

- `1000` 和 `5000` 的受益 task 不再完全乱跳
- spatial 获益任务有部分重合
- pattern 比旧版本更连贯

---

# 7. 通过 / 不通过标准

---

## 7.1 通过标准（满足其中两条即可继续）

如果下面至少满足两条，说明这条线还值得继续：

### 条件 1
anchor corruption 对 final action 和 success 已经有明显影响，  
即 geometry path 从“弱 hint”变成“行为相关信号”。

### 条件 2
`trajectory_decoder` 明确开始训练，  
不再是 dead module。

### 条件 3
`500 / 1000 / 5000` 至少两档优于 baseline，  
且比当前版本的 early signal 更稳定。

### 条件 4
task-level diff pattern 比当前版本更稳定，  
不再完全随机。

---

## 7.2 不通过标准（满足任意一条就高度建议停止）

### 条件 A
修正 coupling 后，anchor corruption 仍然几乎不影响 final action

### 条件 B
`trajectory_decoder` 接上后依然不学，或者输出不起作用

### 条件 C
修正后 early positive signal 也消失，只剩随机波动

### 条件 D
几何路径接入后只带来更大 loss / 更差性能，  
但没有任何机制层面的正信号

如果满足这些情况，就不要继续在这条线上投入更多时间。

---

# 8. 本轮完成后的决策树

---

## 情况 1：机制成立，早期效果也更强

表现为：

- anchor corruption 敏感度明显上升
- decoder 在学
- `500 / 1000 / 5000` 表现更好

### 决策

继续。  
下一步可考虑：

- 更稳的 residual fusion
- 更强的几何约束
- task-level failure mode 分析

---

## 情况 2：机制成立，但性能暂时没涨

表现为：

- anchor corruption 已明显影响行为
- decoder 在学
- 但 early eval 暂时没明显涨

### 决策

仍然可以继续一小步。  
因为这至少说明“几何路径接通了”。  
后续应优先调：

- loss 权重
- decoder 结构
- geometry path 与 action path 的融合强度

---

## 情况 3：机制不成立

表现为：

- anchor corruption 还是几乎无影响
- decoder 还是 dead
- 分数也没更稳定

### 决策

基本停止。  
因为这已经说明：  
即使你显式接通 geometry path，这条线在当前架构里也没有形成有效行为控制力。

---

# 9. 推荐的具体执行顺序

## 第一步
修 final action coupling，  
让 `trajectory_decoder` 真正决定或主导 `pred_action`

## 第二步
确认 decoder 参数可训练、梯度非零、输出非零

## 第三步
只训练到 `500 / 1000 / 5000`

## 第四步
做 anchor corruption test

## 第五步
做 `LIBERO-90` 小评测

## 第六步
整理一张 task-level diff 表

## 第七步
按“通过 / 不通过标准”做 go / no-go 决策

---

# 10. 一句话总结

当前版本最不能说明的问题，不是“几何思路错了”，而是：

**你还没有真正测试到“几何路径决定动作”这件事。**

所以下一轮唯一值得做的实验，就是：

**先把 geometry path 变成 final action 的必要路径，再用最小规模实验验证它是否真的改变行为。**

如果这一步之后，anchor 仍然几乎不影响 action，  
那这条线就非常接近应该停止。

---