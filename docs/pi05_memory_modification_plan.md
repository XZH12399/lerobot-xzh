# Ordered Latent Planner + Low-level FM 方案总结

## 1. 目标

目标是为长程机器人任务设计一套**分层式 VLA / FM 框架**：

- **上层**负责生成一个具有顺序结构的连续潜空间规划序列（ordered latent plan）。
- **下层**负责在当前状态下，结合历史信息和上层规划信息，生成当前 chunk 的具体动作。
- 不使用显式离散子任务标签，而是使用**连续 phase latent** 表达“当前阶段—下一阶段—更远未来”的语义规划。
- 重点解决的问题：
  - 单纯基于历史和当前状态直接生成局部动作，容易缺乏长期一致性。
  - 一次性生成整段 long-horizon action 序列又很难训练、很不稳。
  - 显式子任务划分过硬，不适合连续操作场景。

---

## 2. 总体思路

整体框架分为三层：

1. **状态与记忆编码层**
   - 编码当前 observation、语言指令、历史 memory。
   - 得到当前时刻的记忆增强状态表示。

2. **上层规划层（Planner FM / Ordered Latent Planner）**
   - 基于当前状态和历史，生成一串 ordered latent：
     \[
     Z_t = [z_t^{(1)}, z_t^{(2)}, \dots, z_t^{(K)}]
     \]
   - 这些 latent 不是显式文本子任务，而是连续的 phase-level 规划表示。

3. **下层动作层（Low-level FM）**
   - 利用当前状态作为 query，去读取：
     - 历史 memory
     - 上层 planning latent
   - 再生成当前 chunk 的 action velocity field / action chunk。

---

## 3. 模块划分

---

## 模块 A：Observation + Instruction Encoder

### 作用
编码当前视觉观测和语言指令，得到当前时刻的基础语义表示。

### 输入
- 当前图像 / 观测：
  \[
  o_t
  \]
- 长程语言指令：
  \[
  l
  \]

### 输出
- 感知特征：
  \[
  p_t
  \]
- 当前高层语义特征：
  \[
  c_t
  \]

### 说明
这一层可以沿用类似 MemoryVLA 的思路：
- 用视觉编码器提取 perceptual tokens
- 用 VLM/LLM 得到一个高层 cognitive token

---

## 模块 B：History Memory Bank

### 作用
记录过去发生过的重要信息，供当前时刻查询。

### 输入
- 历史时刻的状态摘要
- 历史时刻的高层语义摘要
- 历史时刻的重要感知特征

### 输出
- 历史 memory token 序列：
  \[
  M_t = [m_1, m_2, \dots, m_H]
  \]

### 说明
这里的 memory 主要承担：
- 记录过去做过什么
- 当前阶段已经推进到哪里
- 某些当前单帧无法判断但历史相关的信息

例如：
- 按钮是否已经按过
- 某个物体是否已经被移动
- 某个目标是否被遮挡过

---

## 模块 C：Memory-enhanced State Aggregator

### 作用
把当前 observation/instruction 表征和历史 memory 融合，形成当前决策所用的统一状态表示。

### 输入
- 当前感知特征：
  \[
  p_t
  \]
- 当前高层语义特征：
  \[
  c_t
  \]
- 历史 memory：
  \[
  M_t
  \]

### 输出
- 当前记忆增强状态：
  \[
  h_t
  \]

### 说明
这一步相当于回答：
- 现在看到什么
- 过去发生过什么
- 当前最关键的状态信息是什么

---

## 模块 D：Ordered Latent Planner（上层规划 FM）

### 作用
生成一串有序的连续规划 latent，用来覆盖整个长程 horizon。

### 输入
- 当前记忆增强状态：
  \[
  h_t
  \]
- 历史摘要（可选）
- VLM 中高层特征（可选）
- planner 的初始噪声/query（如果用 flow matching）

### 输出
- ordered latent plan：
  \[
  Z_t = [z_t^{(1)}, z_t^{(2)}, \dots, z_t^{(K)}]
  \]

其中：
- \(z_t^{(1)}\)：当前最主要 phase
- \(z_t^{(2)}\)：下一阶段 phase
- \(z_t^{(3)}\)：更远 future phase
- ...
- \(z_t^{(K)}\)：最远期的 planning latent

### 关键设计点
这些 latent 不是：
- 离散子任务 id
- 文本指令
- 动作 chunk 的简单压缩码

而应是：
- 连续的
- 有顺序的
- 能表达 phase progression 的 planning latent

---

## 模块 E：Plan Latent Memory / KV Projection

### 作用
将 planner 生成的 \(Z_t\) 投影成下层可查询的 attention memory。

### 输入
- ordered latent plan：
  \[
  Z_t = [z_t^{(1)}, \dots, z_t^{(K)}]
  \]

### 输出
- 对应的 plan keys：
  \[
  K^{plan} = [K_1, \dots, K_K]
  \]
- 对应的 plan values：
  \[
  V^{plan} = [V_1, \dots, V_K]
  \]

其中：
\[
K_i = W_K z_t^{(i)}, \quad V_i = W_V z_t^{(i)}
\]

### 说明
这样做的好处是：
- 下层不需要直接消费整串 z
- 下层可以按需查询不同 future latent
- 未来阶段能对当前动作形成“软约束”

---

## 模块 F：History + Plan Unified Prefix（可选统一接口）

### 作用
把历史 memory 和未来 plan latent 统一成一个可查询前缀。

### 输入
- 历史 memory：
  \[
  M_t = [m_1, \dots, m_H]
  \]
- 未来计划 latent：
  \[
  Z_t = [z_1, \dots, z_K]
  \]

### 输出
- unified prefix：
  \[
  C_t = [m_1,\dots,m_H,\ z_1,\dots,z_K]
  \]

### 推荐增强
给 prefix 中每个 token 加：
- **type embedding**
  - history token
  - plan token
- **position embedding**
  - 表示离当前有多远的过去/未来

### 说明
这使得下层可以自己决定：
- 当前更该关注历史
- 还是更该关注未来规划

---

## 模块 G：Low-level FM（下层动作生成器）

### 作用
根据当前状态、历史信息和上层规划，引导生成当前 chunk 的动作。

### 输入
- 当前状态表征：
  \[
  h_t
  \]
- 当前 query：
  \[
  q_t = Q(h_t)
  \]
- 历史 memory K/V
- plan latent K/V

### 输出
- 当前动作 chunk：
  \[
  a_{t:t+H-1}
  \]
  其中一般：
  \[
  H = 16
  \]

### 推荐做法
下层用 query 去分别读取两类信息：

#### 1. 历史摘要
\[
h_t^{hist} = \text{Attn}(q_t, K^{hist}, V^{hist})
\]

#### 2. 规划摘要
\[
h_t^{plan} = \text{Attn}(q_t, K^{plan}, V^{plan})
\]

#### 3. 融合
\[
g_t = \sigma(\text{MLP}(q_t))
\]
\[
c_t = g_t \odot h_t^{hist} + (1-g_t)\odot h_t^{plan}
\]

#### 4. 动作生成
\[
a_{t:t+H-1} \sim \pi(\cdot \mid h_t, c_t)
\]

### 说明
下层的作用不是做显式规划，而是：
- 读取当前最需要的上下文
- 在局部闭环下生成具体动作 chunk

---

## 4. 数据流 / 连接关系

整体前向流程如下：

### Step 1：编码当前观测和指令
\[
(o_t, l) \rightarrow (p_t, c_t)
\]

### Step 2：融合历史 memory
\[
(p_t, c_t, M_t) \rightarrow h_t
\]

### Step 3：上层 planner 生成 ordered latent plan
\[
h_t \rightarrow Z_t = [z_t^{(1)}, \dots, z_t^{(K)}]
\]

### Step 4：将 \(Z_t\) 投影成 plan K/V
\[
Z_t \rightarrow (K^{plan}, V^{plan})
\]

### Step 5：历史 memory 也投影成 K/V
\[
M_t \rightarrow (K^{hist}, V^{hist})
\]

### Step 6：下层 FM 用当前状态 query 读取历史与规划
\[
q_t = Q(h_t)
\]
\[
h_t^{hist} = \text{Attn}(q_t, K^{hist}, V^{hist})
\]
\[
h_t^{plan} = \text{Attn}(q_t, K^{plan}, V^{plan})
\]

### Step 7：融合上下文，生成当前动作 chunk
\[
c_t = \text{Fuse}(h_t^{hist}, h_t^{plan})
\]
\[
a_{t:t+H-1} \sim \pi(\cdot \mid h_t, c_t)
\]

### Step 8：执行若干动作后，进入下一状态，更新 memory
\[
(o_{t+1}, M_{t+1})
\]

然后重复以上过程。

---

## 5. 关于 planner 的核心语义设定

这里有一个非常关键的原则：

### 不是“一个 z 对应一个离散子任务”
而是：
- 一个 \(z_i\) 对应一个 phase region / phase intent
- 它可能跨越多个 action、多个 chunk
- 它和下一个 \(z_{i+1}\) 之间是连续过渡的，不是硬切换

### 不是“做完一个 chunk 就必须 z 向后移一格”
而是：
- 每次根据当前新状态重新规划出新的 \(Z_t\)
- 但新的 \(Z_t\) 应保持一定顺序结构和阶段持续性

---

## 6. 训练时的基本思路

当前阶段先不管失败过程，最核心的训练目标可以先围绕三件事：

### 1. 下层动作监督
让 low-level FM 学会生成正确的 action chunk。

### 2. 不同 planner latent 的 horizon specialization
让不同 \(z_i\) 负责不同 future horizon，避免所有 \(z_i\) 塌成一样。

### 3. planner 内部的序列结构
保证 \(z_1,z_2,z_3,\dots\) 真的是 ordered latent，而不是多个无关槽位。

---

## 7. 关键注意点

下面这些点是代码落地时最容易踩坑的。

### 注意点 1：不要让 planner latent 退化成“动作切块码”
如果你简单地让：
- \(z_1\) 监督 chunk1
- \(z_2\) 监督 chunk2
- ...

它很容易学成“第 i 个时间块的动作压缩码”，而不是真正规划。

所以必须让 planner latent 同时承载：
- 当前 future chunk
- 更远 future 的可行性/阶段语义

---

### 注意点 2：不要把历史和规划简单裸拼接后不区分
如果把：
\[
[m_1,\dots,m_H,\ z_1,\dots,z_K]
\]
直接裸拼，下层可能把它们都当普通上下文 token，最后历史和规划角色混掉。

推荐做法：
- 加 type embedding
- 或者用分区 attention
- 或者至少分别聚合 history summary 和 plan summary

---

### 注意点 3：planner 不是越长越好
理论上可以把整个长程任务切成很多 latent，但实际上：
- 太少：未来结构不够
- 太多：容易退化成动作级表示

所以 latent 个数 \(K\) 本质上是一个 **planning granularity** 超参数。

---

### 注意点 4：上层 latent 需要顺序结构，不然会退化成“多个 token 但没规划”
即使输出了：
\[
z_1,z_2,z_3,\dots
\]
如果没有顺序建模和训练约束，它们很可能只是多个并列隐藏变量。

---

### 注意点 5：低层是闭环 reactive 的，上层需要更强的结构约束
低层动作每次都被当前真实状态重新锚定，所以不需要跨次显式一致性。  
但上层 planner latent 没有直接标签，容易漂，所以更需要结构约束。

---

### 注意点 6：chunk 边界不等于 phase 边界
一个 phase 可以跨多个 chunk；一个 chunk 内也可能正好发生阶段切换。  
所以不能把：
- planning latent
- action chunk index  
简单一一绑定为绝对对应关系。

---

## 8. 当前版本的建议实现优先级

为了先跑通代码，推荐按这个顺序实现：

### V1：最小可行版本
- 当前 observation encoder
- history memory
- ordered latent planner
- plan K/V projection
- low-level FM query history + plan
- 动作 chunk 监督

### V2：增强版本
- history / plan 分区 attention
- type embedding + temporal embedding
- 不同 \(z_i\) 的 horizon specialization

### V3：更完整版本
- planner 序列连续性约束
- 跨时刻轻量 consistency
- utility / progress critic
- failure process handling

---

## 9. 当前方案的核心优点

### 优点 1
避免了“只看当前和历史做局部动作”的短视问题。

### 优点 2
避免了一次性生成完整长程 action 序列的训练困难。

### 优点 3
用连续 latent 替代显式离散子任务，更适合连续控制。

### 优点 4
让未来任务对当前动作形成反向约束。

例如：
- 后面要用的物体，前面不能藏起来
- 后面要走的路线，前面不能堵死
- 后面要完成的子任务，会影响当前动作如何选择

---

## 10. 当前方案的核心风险

### 风险 1
planner latent 退化成动作压缩码，而不是真规划。

### 风险 2
history 和 plan 混成一个大上下文池。

### 风险 3
不同 \(z_i\) 全塌成一样，失去顺序意义。

### 风险 4
planner 每次重规划都完全改口，没有阶段持续性。

---

## 11. 一句话总结

这套方案的核心是：

**先利用 observation + instruction + history 得到当前记忆增强状态，再由上层 planner 生成一串 ordered continuous plan latents；这些 latents 被投影成 attention K/V，与历史 memory 一起作为统一可查询上下文，由下层 FM 在生成当前 action chunk 时按需读取，从而同时利用过去信息与未来规划约束。**