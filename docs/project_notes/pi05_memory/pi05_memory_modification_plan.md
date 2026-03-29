# PI05 Memory Modification Plan

## 1. 目标

当前更认可的目标不是继续把 `pi05_memory` 仅仅做成一个 history-retrieval baseline，而是把它推进为一套更完整的闭环 latent planning 框架：

- 当前 action chunk 不只依赖当前观测，也依赖过去证据与未来假设。
- 不使用显式离散子任务标签，而是使用连续 latent 表达阶段、进度和未来用途。
- 历史不再优先表示为无限增长的 token 序列，而应表示为可查询、可压缩的历史场。
- 未来不再优先表示为固定长度 planning token list，而应表示为在历史条件下生成的未来场。

这个方向主要想解决三类问题：

- 只看当前和历史做局部动作，容易缺乏长期一致性。
- 一次性生成完整 long-horizon action 序列，训练难、稳定性差。
- 显式离散子任务划分过硬，不适合连续操作场景。

---

## 2. 总体框架

当前更推荐的统一框架是：

1. 编码当前观测、当前状态和长程任务目标。
2. 从历史场 `H_t` 中读取当前相关的过去证据。
3. 基于历史场、当前观测、当前状态和长程目标生成未来场 `F_t`。
4. 当前 chunk 通过 query 同时读取历史场和未来场。
5. 根据读出的上下文生成当前 action chunk。
6. 执行后，根据真实发生结果得到更新 latent，并写回历史场。

也就是说，系统形成一个闭环：

- history field: evidence
- future field: hypothesis
- current chunk: query both
- actual execution: update history field

---

## 3. 当前 chunk 的生成

当前更倾向于不显式维护一个单独的全局摘要向量 `h_t`，而让当前 chunk 的 query 在读取时动态聚合历史和未来信息。

推荐形式：

\[
c_t^{hist} = \mathrm{Read}(q_t, H_t)
\]

\[
c_t^{future} = \mathrm{Read}(q_t, F_t)
\]

\[
a_{t:t+K-1} \sim \pi(\cdot \mid o_t, s_t, c_t^{hist}, c_t^{future})
\]

其中：

- `q_t`：当前 chunk 对应的 query
- `H_t`：历史场
- `F_t`：未来场
- `c_t^{hist}`：从历史场读取的过去证据
- `c_t^{future}`：从未来场读取的未来约束/假设

当前更推荐的理解是：

- 当前 chunk 不是只看近历史，也不是只看近未来。
- 当前 chunk 应自己学习该读历史中的哪一段、未来中的哪一段。
- 过去和未来最好分开读取，再做融合，而不是简单裸拼为一个大上下文池。

---

## 4. 历史场（History Field）

### 4.1 定义

历史场表示为：

\[
H_t(\tau), \quad \tau \in [-T, 0]
\]

其中 `\tau` 是统一真实时间轴上的坐标。

历史场的目标不是保存原始 state/action 序列，而是保存一个：

- 可查询
- 可压缩
- 控制相关
- 与真实交互结果一致

的历史 latent 表示。

### 4.2 历史场里存什么

当前更一致的原则是：

- 不需要长期保存原始 `state/action` 作为查询对象。
- 但历史 latent 的构造必须由真实 `observation/state/action` 驱动。
- 真正被长期保存和查询的是历史 latent，而不是 replay buffer。

因此，原始 `o_t, s_t, a_t` 的角色是：

- 用来生成和更新历史 latent

而不是：

- 在推理阶段被反复直接查询

这样历史场更像一种：

- reusable control cache
- evidence memory

### 4.3 时间不变，变化的是表示密度

分层压缩历史时，真正变化的不是时间本身，而是：

- 这段历史所需的控制点/参数数量

也就是说：

- 平稳段可以用更少参数表示
- 剧烈变化段需要更多参数表示

时间坐标始终绑在真实时间轴上，不会因为压缩而改变物理时间。

---

## 5. 历史更新：应该写入什么

当前更一致的结论是：

- 历史场不应直接写入执行前的 planning latent
- 历史场也不应直接写入原始 observation/state
- 历史场应写入“执行后根据真实结果得到的更新 latent”

这点非常关键，因为历史场记录的是：

- 实际发生后的证据

而不是：

- 执行前的设想

### 5.1 训练时的 posterior latent

训练时可以利用真实 demonstration 动作和真实结果，反推出 hindsight/posterior latent：

\[
z_t^{post} = q_\phi(H_t, o_t, s_t, a_{t:t+K}^{gt}, o_{t+K}, s_{t+K}, g)
\]

然后写入历史场：

\[
H_{t+1} = \mathrm{Write}(H_t, z_t^{post})
\]

这里 `z_t^{post}` 的语义是：

- 对这一段真实交互的 hindsight high-level interpretation

### 5.2 推理时的 execution latent

推理时没有 ground-truth action，但仍然有：

- agent 实际执行的动作
- 执行后的新观测和新状态

因此推理时仍应在执行后得到一个在线更新 latent：

\[
z_t^{exec} = u_\psi(H_t, o_t, s_t, a_{t:t+K}^{exec}, o_{t+K}, s_{t+K}, g)
\]

再写入历史场：

\[
H_{t+1} = \mathrm{Write}(H_t, z_t^{exec})
\]

所以：

- training: write posterior / hindsight latent
- inference: write execution / online update latent

当前更不推荐的做法是：

- 直接把执行前的 `z_t^{plan}` 存成历史

因为那会把 imagined future 污染到 evidence memory。

---

## 6. 未来场（Future Field）

### 6.1 定义

未来更不适合简单表示为固定长度 planning token 序列，而应表示为一个在当前条件下生成的 prospective latent field：

\[
F_t(s), \quad s \in [0, T_f]
\]

它的条件来源包括：

- 历史场 `H_t`
- 当前观测 `o_t`
- 当前状态 `s_t`
- 长程任务目标 `g`

即：

\[
F_t \sim p_\theta(\cdot \mid H_t, o_t, s_t, g)
\]

### 6.2 语义

未来场不是对真实未来的访问，而是：

- 当前条件下生成的未来假设
- 可被下一时刻重新生成和修正

因此：

- history = evidence
- future = hypothesis

### 6.3 当前 chunk 如何使用未来场

当前 chunk 不应被理解为只看“最近未来 token”。

更合理的方式是：

- 给未来场明确的相对时间坐标
- 让 chunk 的 query 自己学习该看哪一段未来
- 不手工固定“越近越重要”

当前更推荐：

- 使用 relative time information
- 使用 attention 自己学习重要性
- 不再强行写死距离权重

如果后续需要加入先验，也更推荐：

- 可学习相对时间偏置
- 或未来置信度 / mutability 信号

而不是手工固定的近远权重函数。

---

## 7. 历史场是否需要单独的全局向量 `h_t`

之前考虑过维护一个独立递推的全局状态 `h_t`，但从“大道至简”的角度，目前更偏向：

- 不单独维护 `h_t`
- 让当前 query 直接从 `H_t` 中完成历史聚合

也就是说，`h_t` 如果需要存在，也更适合作为：

- 对 `H_t(\tau)` 的一个计算性读出

而不是：

- 一个额外维护和训练的独立状态变量

当前更极简的理解是：

- 历史聚合 = query-time readout
- 不必预先压成一个固定全局摘要

---

## 8. 长期压缩：分层 spline 历史场

### 8.1 为什么选择分层 spline

当前最看好的长期历史压缩方式，是在 spline 层面做分层合并：

- 近期历史：高分辨率控制点
- 中期历史：中等分辨率控制点
- 久远历史：低分辨率控制点

这个方向的核心优点是：

- 保留统一时间坐标
- 支持按时间查询
- 支持近密远疏
- 可控制存储增长

### 8.2 为什么不能只是普通 token cache

如果一直把历史保存为 token 列表，历史会线性增长。  
即使这些 token 后面仍然通过 attention 查询，成本和长度仍会持续累积。

分层 spline 的目的就是：

- 不改变时间语义
- 只降低对平稳历史的表示密度

### 8.3 为什么不是简单平均合并 latent

当前更认同在 spline 层面合并，而不是直接对 latent 语义做粗暴平均。

原因是：

- 在曲线参数层合并更自然
- 近似对象是历史轨迹的连续表示
- 不会直接破坏 latent 的局部语义结构

### 8.4 分层合并的原则

当前更看好的原则是：

- 近期：高分辨率，不轻易合并
- 久远：允许更粗压缩
- 平稳段：可以 aggressively 合并
- 剧烈变化段：需要保留更多控制点

所以真正需要的不是“固定每段同样数量的点”，而是：

- 按变化复杂度分配表示精度

### 8.5 一个关键注意点

虽然不同时间位置的控制点密度会不同，但这不意味着时间被扭曲了。

真正不变的是：

- 真实时间坐标 `\tau`

变化的是：

- 用多少参数去近似该时间段的历史曲线

这和“直线只需要两个点，复杂曲线需要更多点”是同一个道理。

---

## 9. 当前版本最小可行结构

当前更认可的极简结构可以总结为：

1. 用真实 `observation/state/action` 驱动生成并更新历史 latent
2. 将历史 latent 压缩为连续历史场 `H_t(\tau)`
3. 基于 `H_t`、当前状态和长程目标生成未来场 `F_t(s)`
4. 当前 chunk query 同时读取 `H_t` 和 `F_t`
5. 根据读取结果生成当前 action chunk
6. 执行后，用真实结果得到 `z_t^{exec}` 或训练时的 `z_t^{post}`，再写回 `H_t`

这使得系统形成一个真正的闭环：

- past evidence -> history field
- current condition -> future field
- current chunk -> query both
- actual execution -> update history field

---

## 10. 仍需继续明确的问题

虽然整体结构已经更清楚，但仍有几个最关键的开放问题：

1. posterior / execution latent 的最小可行参数化方式是什么  
2. 历史场 `Write` 机制应如何定义  
3. 未来场 `p_\theta` 如何训练成真正具有规划属性，而不是低层动作提示  
4. 分层 spline 历史场何时触发合并，以及如何保留关键事件信息  
5. 当前 chunk 对历史场和未来场的读取，是双路 cross-attention，还是统一 readout 后再融合

---

## 11. 当前阶段的一句话结论

当前更认可的方向，不是“history token + ordered future token”，而是：

**将历史表示为可查询、可压缩的连续 evidence field，将未来表示为在历史条件下生成的 prospective field，并让当前 chunk 通过 query 机制同时读取过去证据和未来假设来生成动作。**

---

## 12. 多尺度曲线规划视角

为了避免后续讨论中再次把 latent、subtask、history、future 拆得过碎，当前可以进一步用一个统一的几何视角来理解整套方案：

- 高层规划不是输出离散 subtask 列表，而是在任务潜空间中生成一条连续的主曲线。
- 低层控制不是简单执行主曲线，而是在主曲线的当前局部位置上，结合实时观测和状态，细化出更小尺度、可执行的局部曲线。
- 因而整套系统更像是一个多尺度曲线细化过程，而不是“离散高层命令 + 低层被动执行”。

### 12.1 高层曲线的含义

这里的“曲线”不应理解为机械臂末端在物理空间中的几何轨迹，也不应直接理解为原始 action 序列本身。

更准确地说，它表示的是任务潜空间中的连续规划轨迹，描述的是：

- 当前处于整条任务推进中的哪个阶段
- 已经完成了哪些关键结构
- 当前正在向哪个未来用途或未来阶段过渡
- 后续整体路径会如何继续弯折和展开

因此，显式的 subtask 可以被看作这条连续曲线上的某些局部区域或局部片段，而不是最基本的表示单元。

### 12.2 连续 latent 的含义

在这个视角下，latent 更适合被理解为：

- 曲线上的当前位置
- 曲线在当前位置附近的局部方向
- 当前局部片段所代表的阶段、进度与过渡趋势

所以连续 latent 不是对离散 subtask 的简单 soft label，而是对“当前连续规划状态”的编码。

### 12.3 历史与未来的重新理解

在曲线视角下：

- history field 对应已经走过的曲线前缀
- future field 对应尚未展开的曲线后缀
- 当前 chunk 的生成，本质上是在读取整条曲线中与当前最相关的历史前缀和未来后缀

也就是说，过去和未来并不是两个外加模块，而是同一条连续规划曲线在当前时刻两侧的不同部分。

### 12.4 低层 action chunk 的角色

低层 chunk 不应被理解为“执行一个离散子任务”，而更像是在主曲线当前局部位置上，进一步展开一段更高分辨率的小曲线。

因此：

- 高层给出的是粗尺度的连续任务结构
- 低层负责将其细化为局部可执行轨迹
- 当前 chunk 的 action 由主曲线局部片段、当前观测和当前状态共同决定

这也解释了为什么未来会影响当前动作：即使当前位置接近，如果未来后缀不同，局部细化出来的小曲线也会不同。

### 12.5 与现有方案的对应关系

若沿用当前文档中的记号，那么可以将：

- 历史场 `H_t` 视为曲线前缀的连续表示
- 未来场 `F_t` 视为曲线后缀的连续表示
- 当前 chunk 的 query 视为“从整条曲线中读取当前局部几何上下文”的机制
- writeback latent 视为执行后对曲线前缀的真实增量写入

这样一来，history field / future field / current chunk / writeback 这四部分就不再是彼此割裂的模块，而是同一条连续规划曲线的不同侧面。

### 12.6 当前阶段的一个重要结论

当前更认可的不是“先离散切 subtask，再给每个 subtask 一个 embedding”，而是：

**把高层规划建模为任务潜空间中的连续主曲线，把低层控制建模为对主曲线局部片段的细化。**

这个观点后续可以直接作为历史场、未来场、future update 和 chunk generation 的统一解释框架。
