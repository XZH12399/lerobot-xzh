# pi05_spatial 下一阶段方案：在 latent / denoising 阶段做更强几何约束

> 目标：
>
> 不再走 `post-hoc trajectory_decoder full takeover` 这条已经被证伪的路线，  
> 而是在 **FM / action denoising 主干内部**，让 geometry 成为更强的中间约束。
>
> 核心原则：
>
> - **保留 FM final action branch**
> - **几何信息进入 denoising 过程，而不是事后接管 final control**
> - **几何约束必须比旧版更强，不能再只是几乎无影响的 weak hint**
>
> 当前已有证据表明：
>
> 1. 旧版 soft-coupling 中，anchor corruption 对 final action 影响不到 `1%`，说明几何路径太弱。  
> 2. 强耦合 A-version 中，geometry 已经能明显影响 final action，但行为整体崩掉，说明“post-hoc decoder full takeover”这条实现不对。  
>
> 因此下一阶段最合理的方向是：
>
> **让 geometry 在 latent / denoising 内部成为强条件，而不是让几何 decoder 在最后接管动作。**

---

# 0. 这阶段要解决的真实问题

旧版 `pi05_spatial` 的问题不是“几何没有进模型”，而是：

- 进了，但太软
- 几何信息更像 side hint
- 行为层几乎可以忽略它

而强耦合 A-version 的问题不是“几何不能影响行为”，而是：

- 它影响行为的方式太晚
- 太替代式
- 直接破坏了原本可工作的 FM action policy

所以现在真正该做的不是：

- 再做一个更强的 post-hoc decoder
- 或再做一个更复杂的 trajectory takeover

而是：

> **在 FM 的 action denoising 过程中，把 geometry 变成更强的生成约束。**

---

# 1. 总体设计原则

## 1.1 保留什么

必须保留：

- FM / diffusion / denoising 的主 action 生成路径
- 原始 final action head
- 原始 task-conditioned action semantics
- 原始 gripper / temporal control 生成逻辑

也就是说，最终仍然是：

`fm_hidden -> fm_action_head -> final_action`

而不是：

`fm_hidden -> geometry decoder -> final_action`

---

## 1.2 增强什么

增强的是：

- action token 如何读图
- action token 如何被几何信息调制
- action token 在每层 denoising 中如何被 anchor / spatial feature 约束
- hidden update 时几何分支的存在感

---

## 1.3 避免什么

要避免以下三种重新走偏：

### A. 几何只作为普通 embedding 拼进去
这太弱，容易再次退化成 side hint。

### B. 几何只影响 auxiliary loss
这样行为层仍可能绕开它。

### C. 几何在最后完全 takeover final action
这条线已经应停止。

---

# 2. 新阶段的核心思路

一句话总结：

> **让 geometry 约束 denoising 过程本身，而不是在 denoising 结束后重新生成动作。**

更具体地说：

- 每个 action token 在每层都带着 geometry anchor
- 这个 anchor 不只是做轻量 embedding
- 它要参与：
  - 视觉读取范围
  - 局部特征注入
  - hidden update 强度
  - 残差修正
- 但最终 action 仍由 FM action head 输出

这意味着 geometry 的角色从：

- “后处理解释器”

变成：

- “生成过程中的强条件 / 强约束”

---

# 3. 推荐的三层强约束结构

建议把几何约束设计成三层，从弱到强依次叠加。

---

## 第一层：几何限制 action token 能看什么

这是视觉读取层的约束。

### 目标

不再让 action token 任意从全图读信息，而是：

- 优先读 anchor 附近
- 必要时只允许从 anchor 附近窗口读

### 做法

对 action-to-image 的 attention 或局部 feature readout，加更强的空间限制。

从原来的：

- soft geometry bias

升级为：

- anchor-centered local window
- 或 anchor-centered masked attention

### 形式

不是只有：

`score = qk + bias(anchor, pixel)`

而是进一步改成：

- 只有 anchor 邻域 token 可参与
- 邻域外 token 权重强衰减或直接 mask 掉

### 作用

这一步让 geometry 不再只是“建议你看附近”，而是：

> **你必须主要从这里读。**

---

## 第二层：几何特征必须进入 action hidden update

这是 hidden state 更新层的约束。

### 目标

避免 geometry feature 只是被拼接一下然后被主干忽略。

### 做法

在每个 denoising block 内，把 geometry local feature 作为 action update 的必要输入。

例如：

- geometry branch 先输出一个 `geom_delta_h`
- 然后 action hidden update 写成：

`h_next = h_base + beta * geom_delta_h`

其中：

- `h_base` 来自原 FM block
- `geom_delta_h` 来自 anchor-conditioned spatial readout
- `beta` 不能太小，且最好是可控的固定值或受限门控

### 原则

几何不是“附加描述”，而是：

> **参与 hidden 残差更新。**

### 注意

这里的 `geom_delta_h` 是改 hidden，不是直接改 final action。  
所以它比 old soft hint 强，但比 post-hoc takeover 安全。

---

## 第三层：几何约束 action residual，而不是接管 action

这是行为层的约束。

### 目标

让 geometry 对最终行为有实际影响，但不让它直接替代原 FM action。

### 做法

在 final action head 前，加一个 geometry-conditioned residual head：

`final_action = fm_action + gamma * geom_action_residual`

其中：

- `fm_action`：原 FM head 输出
- `geom_action_residual`：由 latent 中的 geometry-conditioned hidden 预测
- `gamma`：较小但不可忽略，例如 `0.1 ~ 0.3`

### 为什么这样做

它的好处是：

- geometry 对 final behavior 有可测影响
- 但不会像 full takeover 一样把 FM policy 打崩
- 更接近“几何帮助修正动作”，而不是“几何重新发明动作”

### 这一步和 A-version 的本质区别

A-version 是：

`final_action = decoder(trajectory)`

这里建议的是：

`final_action = fm_action + small_geometry_correction`

所以 geometry 是 correction，不是 replacement。

---

# 4. 推荐的模块结构

下面给出一个更贴近实现的结构。

---

## 4.1 输入

- 双视角图像 `I1, I2`
- 文本指令 `T`
- 机器人状态 `s`
- noisy action tokens `z_a`
- diffusion timestep `t_diff`
- geometry anchor `P`

其中：

- `P` 仍可以先用你当前已有的 2D / 3D anchor 表示
- 但它不再承担 final action 解码职责
- 它只负责约束 latent 生成过程

---

## 4.2 编码输出

### 视觉语义流
- `Z_vlm`

### 空间特征流
- `F1, F2`

### action 主干流
- `H_a`

### geometry 状态流
- `P`
- `E_coord = CoordEncoder(P)`

---

## 4.3 每个 denoising block 内的推荐流程

每层 block 建议按下面顺序做。

### Step 1：原始 FM block 得到 base hidden update
`H_base = FMBlock(H_a, Z_vlm, state, t_diff)`

### Step 2：根据 anchor 做局部视觉读取
- 从 `F1, F2` 中按 `P` 采样 local feature
- 或用 local masked attention 读 anchor 邻域

得到：

- `G_local`

### Step 3：geometry branch 生成 hidden residual
`G_delta_h = GeomResidualMLP(H_base, G_local, E_coord, state)`

### Step 4：将 geometry residual 写回 hidden
`H_a_next = H_base + beta * G_delta_h`

这里的 `beta` 建议一开始不要让网络自由学成 0。  
可以先固定一个值，或者限制在较小但非零区间。

### Step 5：更新 anchor（可选）
`P_next = P + delta_P`

这里可以保留 refinement，但它只服务于下一层 geometry readout。

### Step 6：最后 action head 仍走 FM 主头
在最后一层之后：

`fm_action = FMActionHead(H_a_final)`

再加一个较小的 geometry residual：

`geom_residual = GeomActionResidualHead(H_a_final, G_local_final, E_coord_final)`

`final_action = fm_action + gamma * geom_residual`

---

# 5. 三种具体可落地版本

建议按从稳到强的顺序来做。

---

## 5.1 版本 S1：强局部读取 + hidden residual

### 做法

- geometry 只影响 hidden
- final action 仍完全由 FM head 输出

即：

- local masked attention / local sampling
- `H_next = H_base + beta * G_delta_h`
- `final_action = FMActionHead(H_final)`

### 优点

- 最稳
- 最不容易打崩行为
- 最接近你原始“latent alignment”目标

### 缺点

- 可能仍不够强
- 需要靠 diagnostics 验证 geometry 是否真被使用

### 推荐程度

**第一优先。**

---

## 5.2 版本 S2：hidden residual + 小 action residual

### 做法

- 保留 S1 全部
- 最后再加一个较小的 geometry action correction

`final_action = fm_action + gamma * geom_action_residual`

### 优点

- 比 S1 更容易让 geometry 影响行为
- 比 A-version 安全很多

### 缺点

- 需要控制 `gamma`
- 太大可能重新扰乱行为

### 推荐程度

**第二优先。**

如果 S1 仍然太软，再上 S2。

---

## 5.3 版本 S3：局部窗口硬 mask + 小 residual correction

### 做法

- action token 只能从 anchor 附近窗口读局部特征
- hidden update 强依赖 geometry residual
- final 只加小的 geometry correction

### 特点

这是比较强的 latent constraint，但仍不是 final takeover。

### 推荐程度

**第三优先。**

适合在 S1 / S2 已经看到正信号后再用。

---

# 6. 本阶段不要再做的事情

为了防止重新绕回错误路线，下面这些事建议明确不做：

- 不再让 `trajectory_decoder` fully own final action
- 不再把“pred_clean_trajectory -> decoder -> final action”作为主路径
- 不再为了追求耦合强度，牺牲 FM action 分支的主体地位
- 不再把几何分支主要训练成“另一个动作生成器”

这阶段的关键词是：

**constraint**
而不是  
**replacement**

---

# 7. 训练目标设计

---

## 7.1 主目标不变：action loss 仍是核心

必须保持：

- `L_action` 是主目标

因为你最终要优化的是 policy behavior，不是几何自洽性。

---

## 7.2 几何相关损失作为辅助，但不能压过 action

保留但适当减弱：

- `L_coord`
- `L_proj`
- `L_smooth`

原则是：

> geometry loss 的职责是让 geometry branch 对 action 有用，  
> 不是让 geometry branch 自己看起来很漂亮。

---

## 7.3 新增一个“geometry usage”诊断型正则（可选）

如果你发现 geometry 又开始被忽略，可加一个轻量 usage regularizer。

例如约束：

- `geom_action_residual` 不能长期趋近全 0
- `G_delta_h` 范数不能长期塌缩
- local attention 不能完全无视 anchor 邻域

注意这里只建议做轻量监控或轻量正则，  
不要把它变成一个新的大 loss 主项。

---

# 8. 这一阶段最关键的诊断实验

这部分比多跑 benchmark 更重要。

---

## 8.1 诊断 A：anchor corruption test 继续保留

这是标准体检项，必须每版都做。

### 目标

看 geometry 是否真的影响行为。

### 你要追求的状态

- 不再像旧版那样 `<1%`
- 但也不必像 A-version 一样靠 full takeover 才有影响

理想状态是：

> geometry 对 final action 有中等幅度影响，  
> 且这种影响不会导致行为整体崩坏。

---

## 8.2 诊断 B：看 corruption 后是“合理退化”还是“整体崩坏”

和以前不同，这次不只看 action diff ratio，  
还要看 corruption 后失败方式。

### 理想现象

- 正常 anchor 时能完成任务
- shuffle / const anchor 后，局部定位和接近变差
- 但不是整个 policy 完全失去 task awareness

这说明 geometry 在修正局部行为，而不是替代整个 policy。

---

## 8.3 诊断 C：看 hidden 层 geometry residual 是否非零且稳定

记录：

- `||G_delta_h||`
- 各层 `beta * G_delta_h` 的幅度
- 是否长期塌成 0
- 是否不同 task 有差异

### 理想现象

- geometry residual 非零
- 早期稍大，中后期逐渐稳定
- 对不同样本有分辨性

---

## 8.4 诊断 D：task-level diff table

仍然建议在：

- `500`
- `1000`
- `5000`

做 task-level diff。

因为现在你最想知道的是：

- latent constraint 强化后，是否重新出现 early positive signal
- 受益 task 是否比旧版更稳定

---

# 9. 本阶段最小实验闭环

不要一次做太多版本。  
建议按下面顺序推进。

---

## 第一轮：S1

配置：

- 强局部读取
- hidden residual
- final action 仍纯 FM head

评测：

- `500 / 1000 / 5000`
- `LIBERO-90`
- anchor corruption test
- hidden residual logging

### 判断

如果 S1 就已经：

- 比旧 soft version 更依赖 geometry
- 且不打崩行为
- early signal 不差于旧版

那说明方向对。

---

## 第二轮：S2

只有在 S1 太软时才做。

配置：

- S1 全部保留
- final action 加小的 geometry residual correction

评测同上。

### 判断

如果 S2 比 S1：

- corruption sensitivity 更合理
- early eval 更稳
- 行为仍不崩

那可以把它作为下一阶段主版本。

---

# 10. go / no-go 标准

---

## 10.1 值得继续的信号

满足其中两条即可继续：

1. relative to old soft version，anchor corruption sensitivity 明显上升  
2. 没有出现 A-version 那种 behavior collapse  
3. `500 / 1000 / 5000` 至少两档不弱于旧版 early signal  
4. geometry residual / local readout 在 hidden 中稳定非零  
5. task-level diff 比旧版更稳定、更可解释

---

## 10.2 需要警惕的信号

如果出现下面情况，要小心：

1. geometry 还是几乎不影响行为  
2. 一增强 latent coupling 就开始明显打崩 policy  
3. hidden residual 很快塌成 0  
4. corruption 只导致整体乱动，而不是局部能力退化  
5. 所有收益仍只是随机波动

---

## 10.3 应该停止的条件

如果 S1 / S2 都出现下面模式，就应该认真考虑停止这条线：

- geometry 在 latent 中仍然很难形成稳定行为影响
- 一旦增强影响，policy 就明显失稳
- task-level 上始终没有稳定收益模式

那就说明：  
在当前 pi05 / FM 架构里，这条 geometry alignment 路线的结构收益非常有限。

---

# 11. 一句话总结

上一阶段的结论已经很清楚：

- **weak hint 太弱**
- **full takeover 太硬**

所以下一阶段正确的方向是：

> **保留 FM final action branch，  
> 在 latent / denoising 阶段把 geometry 做成更强的中间约束与残差修正，  
> 而不是让 geometry 在最后重新生成动作。**

这才是最贴近你最初研究问题的方案。

---