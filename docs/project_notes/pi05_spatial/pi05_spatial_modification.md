# 基于 LeRobot 的 pi0.5 的 action-image 几何对齐改造方案

## 1. 文档目标

本文档对原方案进行**方向修正**：

- **不再把“主相机坐标系下的共享 3D 轨迹”作为必须替换原 action token 的唯一主线**；
- **保留 LeRobot / pi0.5 原有的 action token 去噪与生成逻辑**；
- **重点修改 action token 与图像 / 指令 latent 融合的方式**，使这种融合不再只是语义相关，而是带有**真实几何坐标对齐**；
- **在训练和推理过程中，为每个 action token 维护一个显式坐标锚点**，并利用该锚点约束 action token 如何读取图像 latent 与空间特征。

更准确地说，本方案的目标不是“完全放弃 action token，改为纯轨迹 diffusion”，而是：

> **在 pi0.5 现有的 action 生成框架中，为 action token 引入一个显式几何坐标锚点，并在 action token 与 image latent 融合时，用这个锚点建立真实坐标对齐。**

因此本文档给出的最终思路是：

1. **保留原有 VLM token 与 action token 的去噪主干**；
2. **为每个 action token 关联一个真实坐标状态**；
3. **在 action token 查询图像条件时，加入几何偏置或局部采样机制**；
4. **使 action token 读到的是“与自己当前几何位置对应”的视觉信息，而不是无约束的全图 latent**。

---

## 2. 核心问题：pi0.5 中真正缺的不是“更多 token”，而是“对齐机制”

### 2.1 原始 pi0.5 的抽象流程

在 LeRobot / pi0.5 风格模型中，可将动作生成主链路抽象为：

1. 双视角图像和文本指令进入 VLM；
2. VLM 输出语义 latent tokens；
3. 动作 expert head 对 noisy action tokens 进行去噪；
4. action token 在 block 内通过 Q 查询 VLM token 的 KV cache；
5. 最终输出 action。

抽象写法为：

\[
Z_{vlm} = \mathrm{VLM}(I^{(1)}, I^{(2)}, T)
\]

\[
\hat{a}_{1:H} = f_\theta(z^a_t, Z_{vlm}, s, t_{diff})
\]

其中：

- \(I^{(1)}, I^{(2)}\)：双视角图像；
- \(T\)：文本指令；
- \(s\)：机器人状态；
- \(z^a_t\)：扩散时间步 \(t\) 下的 noisy action token；
- \(\hat{a}_{1:H}\)：未来 \(H\) 步动作。

### 2.2 原方法的主要问题

原始设计的关键问题不是“没有图像条件”，而是：

> **action token 在查询图像 / 指令 latent 时，没有强的真实几何锚定。**

也就是说：

- VLM token 提供了语义信息；
- action token 也能从 VLM token 中读取信息；
- 但 action token 本身通常只是“第 \(t\) 步动作”的抽象 token；
- 它没有显式表达“我当前对应末端的哪个物理位置”，也没有表达“我应该主要看图像上的哪一块区域”。

因此它与图像 latent 的融合更像是：

- **语义相关的 latent 融合**，而不是
- **真实坐标对齐后的条件融合**。

这会导致：

1. 图像和动作虽然在 latent 中交互，但不一定物理对齐；
2. 对边缘、接触点、抓取位姿等高精度几何关系不敏感；
3. 模型容易走“全局语义 shortcut”，而不是学会“我该看哪儿、我该如何根据当前位置修正动作”。

---

## 3. 修正后的总目标

### 3.1 一句话描述

> **保留 pi0.5 的 action token 去噪主干，但给每个 action token 增加一个显式坐标锚点，并在 action token 融合图像 / 指令 latent 时，用该锚点约束其关注的视觉区域，从而实现几何对齐的 action-image 融合。**

### 3.2 与旧版文档相比的关键修正

旧版文档的主线是：

- 将主去噪变量改成共享 3D 轨迹；
- 双视角只作为约束；
- 最终输出 3D 轨迹。

本次修正后，主线改为：

- **主去噪变量仍然可以是 action token（与 pi0.5 兼容）**；
- **新增一个与 action token 同步维护的显式坐标状态 \(p_t\)**；
- action token 与图像 latent 的融合时，利用 \(p_t\) 做几何对齐；
- 如有需要，\(p_t\) 可以是 2D 或 3D；
- 轨迹 / 3D 建模可作为增强项，而不是强制替换整个 action 表示。

因此新的问题定义不再是：

> “要不要把整个模型改成共享 3D 轨迹 diffusion？”

而是：

> “如何在保留 action token 生成逻辑的同时，让 action token 在读取 image latent 时与真实坐标对齐？”

---

## 4. 新的变量定义

### 4.1 输入变量

- \(I^{(1)}\)：主相机图像；
- \(I^{(2)}\)：副相机图像；
- \(T\)：文本指令；
- \(s\)：机器人状态；
- \(t_{diff}\)：扩散时间步；
- \(z^a_t\)：noisy action tokens。

### 4.2 语义条件变量

- \(Z_{vlm}\)：VLM 输出的语义 token 序列。

### 4.3 新增的几何锚点变量

为每个未来时间步的 action token 维护一个显式坐标状态：

\[
p_t
\]

对于 horizon 为 \(H\) 的未来动作，记：

\[
P = \{p_1, p_2, \dots, p_H\}
\]

每个 action token \(z^a_t\) 不再只是“第 \(t\) 步动作 token”，而变为：

\[
(z^a_t, p_t)
\]

也就是说：

- \(z^a_t\)：表示动作语义 / 控制潜变量；
- \(p_t\)：表示这个动作 token 当前对应的真实几何位置。

### 4.4 坐标的两种可选定义

根据你当前数据条件，\(p_t\) 可以有两种主定义方式。

#### 方案 A：图像平面坐标锚点（更简单）

单视角：

\[
p_t = (u_t, v_t)
\]

双视角：

\[
p_t = (u_t^{(1)}, v_t^{(1)}, u_t^{(2)}, v_t^{(2)})
\]

适合：

- 当前没有稳定 3D 几何恢复；
- 想先做最小侵入式几何对齐。

#### 方案 B：主相机坐标系下的 3D 锚点（更强）

\[
p_t = (x_t, y_t, z_t)
\]

然后投影到双视角：

\[
\Pi_1(p_t), \Pi_2(p_t)
\]

适合：

- 训练数据能恢复出末端 3D 位置；
- 相机标定较可靠；
- 想进一步增强几何一致性。

**本文档主推荐路线**：

> **框架设计先按“一般化坐标锚点 \(p_t\)”来写；工程落地时优先从 2D 锚点开始，后续可平滑升级到 3D 锚点。**

---

## 5. 真实对齐到底是什么意思

“对齐”不是一句抽象口号，而是下面这三层结构同时成立。

### 5.1 第一层：action token 自己要带坐标

原始 action token 只有语义，没有几何位置。现在需要：

\[
q_t^{act} = \mathrm{Fuse}(z_t^a, \phi(p_t), e_t^{time}, e_{diff}, e_s)
\]

其中：

- \(z_t^a\)：原动作 token；
- \(\phi(p_t)\)：坐标编码；
- \(e_t^{time}\)：动作序列时间步编码；
- \(e_{diff}\)：扩散时间步编码；
- \(e_s\)：机器人状态编码。

这一步的作用是：

> **让 action token 不再只是“第 \(t\) 步动作”，而是“第 \(t\) 步、且当前对应坐标 \(p_t\) 的动作”。**

### 5.2 第二层：action token 查询 image latent 时要有几何偏置

假设图像 token 为：

\[
\{k_i, v_i\}_{i=1}^N
\]

第 \(i\) 个图像 token 对应图像上的一个位置 \(c_i\)。原始 attention score 为：

\[
s_{ti} = q_t^\top k_i
\]

现在改成：

\[
s_{ti} = q_t^\top k_i + b(p_t, c_i)
\]

其中：

- \(p_t\)：action token 当前锚点；
- \(c_i\)：image token 对应的位置；
- \(b(p_t, c_i)\)：几何对齐偏置。

最简单的偏置写法：

\[
b(p_t, c_i) = -\lambda \lVert \Pi(p_t) - c_i \rVert^2
\]

解释：

- 距离越近，偏置越大；
- action token 更倾向于读取离自己投影位置更近的图像 token；
- 这样 attention 就从“全局任意查”变成了“优先查几何相关区域”。

如果 \(p_t\) 本身就是 2D 图像坐标，则 \(\Pi(p_t)\) 可直接省略。

### 5.3 第三层：除了 bias，还可直接做局部采样

仅靠几何 bias 还不够强。更直接的方式是：

> **根据 \(p_t\) 在图像上对应的位置，从 spatial feature map 中直接采样局部视觉特征。**

记图像特征图为：

\[
F \in \mathbb{R}^{C \times H_f \times W_f}
\]

则局部采样为：

\[
f_t^{local} = \mathrm{sample}(F, \Pi(p_t))
\]

再将其写回 action token：

\[
q_t^{act} \leftarrow q_t^{act} + W_f f_t^{local}
\]

或者：

\[
q_t^{act} \leftarrow \mathrm{MLP}([q_t^{act}, f_t^{local}, \phi(p_t)])
\]

这一步的意义非常关键：

> **不再让 action token 自己“猜该看哪里”，而是明确根据当前真实坐标，从图像中读取对应位置的视觉证据。**

---

## 6. 为什么这比“把坐标也编码成普通 token 再拼进去”更合理

一个常见但不推荐的做法是：

\[
p_t \rightarrow \mathrm{MLP} \rightarrow z_t^{coord}
\]

再将 \(z_t^{coord}\) 与图像 token、指令 token、action token 一起拼接。

这个问题在于：

1. 坐标被压缩成普通 latent；
2. 多层 attention 后几何含义容易被语义混合淹没；
3. 无法保证高精度几何锚定；
4. 最后还是退回到“坐标信息只是语义相关的一部分”。

因此本文档坚持的原则是：

> **坐标可以被编码，但不能只被编码；原始坐标必须作为显式状态变量一路保留，并继续参与注意力偏置和局部采样。**

所以必须同时保留：

- \(p_t\)：显式坐标状态；
- \(\phi(p_t)\)：坐标 embedding；
- \(\Pi(p_t)\)：用于图像对齐和局部采样的投影位置。

---

## 7. action 坐标从哪里来

这是这个方案最关键的工程问题之一。

### 7.1 训练时：使用 GT 坐标锚点

训练时最简单，因为你拥有数据监督。

#### 如果有 3D 末端轨迹

若数据集中能恢复未来末端位置，则可构造：

\[
p_t^* = (x_t, y_t, z_t)
\]

#### 如果只有图像空间可用

也可构造：

\[
p_t^* = (u_t^{(1)}, v_t^{(1)}, u_t^{(2)}, v_t^{(2)})
\]

无论 2D 还是 3D，本质都是：

> **训练时给每个 action token 一个 GT 几何锚点。**

于是训练过程中的 token 不再是单独的 \(z_t^a\)，而是：

\[
(z_t^a, p_t^*)
\]

### 7.2 推理时：如何获得坐标锚点

推理时没有 GT 坐标，需要模型维护一个当前估计锚点 \(\hat{p}_t\)。

最推荐的做法有两类。

#### 方案 A：迭代维护坐标锚点

每层 block 后，模型预测一个坐标增量：

\[
\Delta p_t = h_p(q_t)
\]

然后更新：

\[
\hat{p}_t \leftarrow \hat{p}_t + \Delta p_t
\]

下一层再用更新后的 \(\hat{p}_t\) 去做对齐。

#### 方案 B：用初始 anchor + 逐层 refinement

在推理开始时，为每个时间步初始化一个坐标锚点 \(\hat{p}_t^{(0)}\)，来源可以是：

- 上一帧估计到的未来轨迹；
- 简单线性外推；
- 单独一个粗轨迹 head；
- 常量模板轨迹。

然后后续层迭代修正。

这通常比“从零开始猜每个点”更稳。

---

## 8. 新方案中的图像侧表示：语义 token 与空间 feature map 分工

### 8.1 保留原有 VLM token 的原因

pi0.5 中 VLM token 很有价值，它们可以继续承担：

- 文本指令语义；
- 场景全局上下文；
- 操作对象身份；
- 粗粒度目标选择。

所以原有：

\[
Z_{vlm} = \mathrm{VLM}(I^{(1)}, I^{(2)}, T)
\]

这条路径不建议删掉。

### 8.2 但 VLM token 不足以承担精细几何对齐

因为最终 VLM token 更偏语义聚合表示，而你现在最需要的是：

- 某个 action token 当前应该看图上哪个位置；
- 该位置局部边缘、把手、接触区具体长什么样；
- action token 当前锚点附近的视觉几何证据。

所以需要再额外保留一条**spatial feature map 支路**：

- \(F^{(1)}\)：主视角空间特征图；
- \(F^{(2)}\)：副视角空间特征图。

### 8.3 分工总结

- **VLM token**：语义条件；
- **spatial feature map**：局部几何证据；
- **action token + 坐标锚点**：需要被条件化的动作状态。

---

## 9. geometry-aware action-image 融合机制

这一部分是本文档的核心。

### 9.1 原始 pi0.5 的融合方式

你已经明确指出：pi0.5 里更接近下面的机制：

- 去噪 token 与 VLM token 在同一套注意力框架中交互；
- 动作 expert token 使用自身的 Q 查询 VLM 的 KV cache。

因此本文方案中：

> **保留原始的 VLM KV cache 条件注入方式；只是在 action token 查询图像条件时，额外加入“坐标对齐”分支。**

### 9.2 融合机制的三部分

对于第 \(t\) 个 action token，记其当前 token 为 \(q_t\)，当前坐标锚点为 \(p_t\)。

#### 部分 A：原有语义条件读取

保留原始逻辑：

\[
q_t \leftarrow \mathrm{AttnToVLM}(q_t, Z_{vlm})
\]

#### 部分 B：几何偏置 attention

对图像 token 加位置相关的偏置：

\[
s_{ti} = q_t^\top k_i + b(p_t, c_i)
\]

这里 \(c_i\) 是图像 token 所在位置。

#### 部分 C：局部采样注入

根据 \(p_t\) 投影到图像上，采样局部特征：

\[
f_t^{(1)} = \mathrm{sample}(F^{(1)}, \Pi_1(p_t))
\]

\[
f_t^{(2)} = \mathrm{sample}(F^{(2)}, \Pi_2(p_t))
\]

再融合：

\[
q_t \leftarrow \mathrm{Fuse}(q_t, f_t^{(1)}, f_t^{(2)}, \phi(p_t))
\]

### 9.3 为什么几何偏置 + 局部采样最好一起用

如果只用几何偏置：

- attention 仍然是“在一堆 patch token 中软选择”；
- 对局部像素级几何约束还不够强。

如果只用局部采样：

- 虽然局部信息更强，但少了对全局 patch token 的几何先验约束。

因此推荐组合：

1. **几何偏置 attention**：限制 action token 优先关注正确区域；
2. **局部采样**：提供当前锚点附近更高精度的局部证据。

---

## 10. 坐标编码与坐标更新

### 10.1 坐标编码

对每个坐标锚点 \(p_t\)，做连续坐标编码。

#### 若为 2D 坐标

\[
p_t = (u_t, v_t)
\]

可用 Fourier embedding：

\[
\phi(p_t) = [
\sin(\omega_1 u_t), \cos(\omega_1 u_t),
\sin(\omega_1 v_t), \cos(\omega_1 v_t),
\dots]
\]

#### 若为 3D 坐标

\[
p_t = (x_t, y_t, z_t)
\]

则：

\[
\phi(p_t) = [
\sin(\omega_1 x_t), \cos(\omega_1 x_t),
\sin(\omega_1 y_t), \cos(\omega_1 y_t),
\sin(\omega_1 z_t), \cos(\omega_1 z_t),
\dots]
\]

再经 MLP：

\[
e_t^{coord} = \mathrm{MLP}(\phi(p_t))
\]

### 10.2 坐标更新

每个 block 结束前，模型预测坐标残差：

\[
\Delta p_t = h_p(q_t)
\]

更新：

\[
p_t \leftarrow p_t + \Delta p_t
\]

这一步是几何对齐机制成立的关键，因为：

- action token 不是绑定一个固定坐标；
- 它会随着去噪过程不断修正“自己现在对应的真实位置”；
- 下一层 block 再根据新位置做更准确的图像读取。

因此整个过程是：

> **坐标感知的 action token 迭代 refinement。**

---

## 11. 双视角情况下的对齐方式

### 11.1 若采用 2D 双视角坐标锚点

则：

\[
p_t = (u_t^{(1)}, v_t^{(1)}, u_t^{(2)}, v_t^{(2)})
\]

此时：

- 对主视角图像 token 的几何偏置使用 \((u_t^{(1)}, v_t^{(1)})\)；
- 对副视角图像 token 的几何偏置使用 \((u_t^{(2)}, v_t^{(2)})\)；
- 对两视角局部采样也分别使用各自坐标。

### 11.2 若采用 3D 锚点

则：

\[
p_t = (x_t, y_t, z_t)
\]

投影到两个视角：

\[
c_t^{(1)} = \Pi_1(p_t), \qquad c_t^{(2)} = \Pi_2(p_t)
\]

再分别对两个视角做：

- 几何偏置 attention；
- 局部采样。

### 11.3 双视角在这里的真正角色

双视角不是为了最终输出两条独立轨迹，而是为了：

- 让同一个 action token 从两个观测视角读取几何证据；
- 减少遮挡歧义；
- 使 action token 对自己对应位置的视觉理解更稳定。

---

## 12. 面向 LeRobot / pi0.5 的最小改法

这是最关键的工程部分：怎样在不大改主干的情况下接进去。

### 12.1 保留不动的部分

以下部分尽量不改：

- VLM 编码图像与文本；
- action token 的 diffusion / denoising 框架；
- action token 查询 VLM KV cache 的逻辑；
- 原有时间步嵌入、状态嵌入、调度方式。

### 12.2 新增的最小模块

#### 模块 1：`ActionCoordinateState`

作用：

- 为每个 action token 维护当前坐标锚点 \(p_t\)；
- 负责初始化、更新和缓存。

#### 模块 2：`CoordinateEncoder`

作用：

- 将 \(p_t\) 编码为 \(e_t^{coord}\)；
- 为 action token 提供位置感。

#### 模块 3：`GeometryBiasAttention`

作用：

- 在 action token 查询 image tokens 时，加入几何偏置项 \(b(p_t, c_i)\)。

#### 模块 4：`ProjectAndSample`

作用：

- 若 \(p_t\) 为 3D，投影到双视角；
- 若 \(p_t\) 为 2D，则直接使用；
- 从 spatial feature map 中做双线性采样。

#### 模块 5：`CoordinateRefiner`

作用：

- 根据当前 action token 与局部视觉证据，预测坐标残差 \(\Delta p_t\)。

### 12.3 替换动作头的建议

不是彻底替换原 action expert head，而是将其改为：

> **geometry-aware action expert head**

即在原 action expert head 内部增加：

1. 坐标状态输入；
2. 几何偏置 attention；
3. 局部采样回写；
4. 坐标 refinement 分支。

---

## 13. block 级别的推荐流程

对每个 denoising block，建议如下。

### Step 1：原始 action token 自注意力 + VLM 条件读取

\[
Q \leftarrow \mathrm{SelfAttnWithVLMCache}(Q, Z_{vlm})
\]

### Step 2：根据当前锚点位置，做几何偏置 attention

对于 image token 位置 \(c_i\)：

\[
s_{ti} = q_t^\top k_i + b(p_t, c_i)
\]

### Step 3：根据当前锚点位置做双视角局部采样

\[
f_t^{(1)} = \mathrm{sample}(F^{(1)}, \Pi_1(p_t))
\]

\[
f_t^{(2)} = \mathrm{sample}(F^{(2)}, \Pi_2(p_t))
\]

### Step 4：融合局部几何证据

\[
e_t^{coord} = \mathrm{MLP}(\phi(p_t))
\]

\[
e_t^{geom} = \mathrm{MLP}([e_t^{coord}, f_t^{(1)}, f_t^{(2)}, e_s, e_{diff}])
\]

\[
q_t \leftarrow \mathrm{Fuse}(q_t, e_t^{geom})
\]

### Step 5：预测 action 更新和坐标更新

- action 分支输出：\(\Delta z_t^a\) 或直接 action 残差；
- coordinate 分支输出：\(\Delta p_t\)。

\[
p_t \leftarrow p_t + \Delta p_t
\]

下一层 block 再基于更新后的 \(p_t\) 做对齐。

---

## 14. 张量定义建议

设：

- batch size 为 \(B\)；
- future horizon 为 \(H\)；
- VLM token 数为 \(N\)；
- hidden dim 为 \(D\)。

### 14.1 输入张量

- `vlm_tokens`: \([B, N, D]\)
- `action_tokens`: \([B, H, D]\)
- `state`: \([B, d_s]\)
- `coord_state`: 
  - 若 2D 双视角：\([B, H, 4]\)
  - 若 3D：\([B, H, 3]\)
- `F1_low / mid / high`: \([B, C_i, H_i, W_i]\)
- `F2_low / mid / high`: \([B, C_i, H_i, W_i]\)

### 14.2 中间张量

- `coord_emb`: \([B, H, D]\)
- `feat1`, `feat2`: \([B, H, C]\)
- `geom_emb`: \([B, H, D]\)
- `delta_coord`: 与 `coord_state` 同形状

### 14.3 输出张量

- `pred_action`: 与原 pi0.5 action 输出同形状
- `pred_coord`: 与 `coord_state` 同形状

---

## 15. 训练目标设计

现在的训练目标不是“只监督最后 action”，而是要同时监督：

1. action 本身；
2. action token 的几何锚点；
3. 几何对齐的一致性。

### 15.1 action 主损失

保持与原 pi0.5 一致：

\[
\mathcal{L}_{action}
\]

若是 diffusion 训练，则是 action noise prediction loss；若是直接回归，则是 \(L_1/L_2\)。

### 15.2 coordinate supervision loss

对每个 action token 关联的坐标锚点监督：

\[
\mathcal{L}_{coord} = \sum_{t=1}^H \lVert \hat{p}_t - p_t^* \rVert_1
\]

其中 \(p_t^*\) 为训练时构造的 GT 坐标锚点。

### 15.3 几何 attention / 采样一致性损失（可选）

若采用 3D 坐标，可加投影监督：

\[
\mathcal{L}_{proj}^{(1)} = \sum_t \lVert \Pi_1(\hat{p}_t) - p_t^{(1)*} \rVert_1
\]

\[
\mathcal{L}_{proj}^{(2)} = \sum_t \lVert \Pi_2(\hat{p}_t) - p_t^{(2)*} \rVert_1
\]

### 15.4 平滑性损失（可选）

如果 \(p_t\) 是连续轨迹点，可加入：

\[
\mathcal{L}_{smooth} = \sum_{t=2}^{H-1} \lVert \hat{p}_{t+1} - 2\hat{p}_t + \hat{p}_{t-1} \rVert_1
\]

### 15.5 总损失

推荐形式：

\[
\mathcal{L} = \mathcal{L}_{action} + \lambda_1 \mathcal{L}_{coord} + \lambda_2 \mathcal{L}_{proj} + \lambda_3 \mathcal{L}_{smooth}
\]

起步建议：

- \(\lambda_1 = 0.5 \sim 1.0\)
- \(\lambda_2 = 0.2 \sim 0.5\)
- \(\lambda_3 = 0.05 \sim 0.1\)

---

## 16. 为什么这套方案更符合你当前真正想做的事

你重新思考后的目标其实是：

> **不是把模型完全改成“轨迹生成器”，而是在 action 生成过程中，让 action token 和图像 latent 的融合发生在真实几何约束下。**

这个目标下，最重要的不是“换输出空间”，而是：

1. 给 action token 一个显式坐标锚点；
2. 让 action token 查询图像时，不再是全局无约束查；
3. 让图像局部几何证据能够直接影响 action token 的更新。

因此这版修正方案相比旧版方案，有三点优势：

### 16.1 对 pi0.5 主干更友好

不需要立刻推翻原有 action diffusion 主逻辑。

### 16.2 更贴近你真正的问题

你最关心的是“融合时如何对齐”，不是“输出变量到底叫轨迹还是 action”。

### 16.3 更利于最小可行实验

你可以先做：

- action token + coordinate embedding；
- 几何 bias attention；
- 局部采样回写；
- coordinate supervision。

这样比直接大改成共享 3D 轨迹 diffusion 的风险更低。

---

## 17. 推荐实验路线

### Stage 0：原始 pi0.5 baseline

不改任何几何对齐模块，作为基线。

### Stage 1：只加 action token 坐标 embedding

\[
q_t \leftarrow q_t + \phi(p_t)
\]

验证“仅让 token 带位置感”是否有提升。

### Stage 2：加 geometry bias attention

\[
s_{ti} = q_t^\top k_i + b(p_t, c_i)
\]

验证“约束 action token 读取正确视觉区域”是否有效。

### Stage 3：加局部采样回写

\[
f_t^{local} = \mathrm{sample}(F, \Pi(p_t))
\]

\[
q_t \leftarrow \mathrm{Fuse}(q_t, f_t^{local}, \phi(p_t))
\]

验证更强显式几何对齐的效果。

### Stage 4：加可学习 coordinate refinement

\[
p_t \leftarrow p_t + \Delta p_t
\]

验证“action token 与坐标锚点共同迭代 refinement”是否进一步提升。

### Stage 5：若效果显著，再升级到 3D 锚点

若 2D 对齐方案已经稳定，再把 \(p_t\) 从 2D 双视角坐标升级为 3D 主相机坐标。

---

## 18. 推荐的第一版 MVP

如果你现在要基于 LeRobot 的 pi0.5 做一个最小可行版本，我最推荐：

### 输入

- 双视角 RGB 图像；
- 文本指令；
- 当前机器人状态；
- noisy action tokens；
- 每个 action token 对应的训练时 GT 坐标锚点。

### 保留

- 原 pi0.5 的 VLM 编码；
- 原 KV cache 条件注入；
- 原 action token 去噪主干。

### 新增

1. `CoordinateEncoder`
2. `GeometryBiasAttention`
3. `SpatialFeatureExtractor`
4. `ProjectAndSample`
5. `CoordinateRefiner`

### 每个 block 中新增

1. action token 基于当前坐标做 geometry-aware attention；
2. action token 根据当前坐标从双视角 feature map 中采样局部特征；
3. 局部特征回写 action token；
4. 预测坐标残差并更新坐标锚点。

### 训练损失

- 原 action loss；
- coordinate supervision；
- 可选 projection / smoothness loss。

---

## 19. 最终结论

本次修正后的方案，不再主张“一步到位把 pi0.5 改成共享 3D 轨迹 diffusion 模型”，而是更贴近你当前真正的目标：

> **在 pi0.5 原有 action token 生成框架中，引入显式几何坐标锚点，使 action token 在与图像 latent 融合时能够与真实坐标对齐。**

因此，真正应该修改的不是“是否还用 action token”，而是下面这件事：

> **让每个 action token 不再是漂浮在潜空间中的抽象动作符号，而是一个绑定了真实位置锚点、能依据该锚点读取局部视觉证据的 geometry-aware action token。**

这意味着：

1. **保留原有 pi0.5 主干逻辑**；
2. **给 action token 加显式坐标状态**；
3. **在 action 查询 image latent 时加入几何 bias**；
4. **同时引入基于当前坐标的局部图像特征采样**；
5. **通过 coordinate supervision 和 refinement 让几何对齐真正成立。**

如果后续实验显示这条路线有效，再进一步升级到：

- 3D 锚点；
- 共享几何轨迹变量；
- 轨迹到 action 的显式两阶段结构。

但第一步，最值得做的，就是把 **action-image 融合从“纯语义 latent 融合”改造成“带真实坐标约束的几何对齐融合”。**
