下面这份内容你可以直接发给 Codex。为了避免 Markdown 再次渲染断裂，我这次不用嵌套代码块，只给你一份纯文本任务说明。

==================================================
任务目标
==================================================

在你已经复制好的 policy `pi05_memory` 上，实现一个最小版的 MemoryVLA-style history memory baseline。

重要原则：
1. 不要完整复刻 MemoryVLA
2. 不要改动 LeRobot pi0.5 原有的下层 FM action head
3. 不要实现 future planner / ordered latent planner
4. 只实现“显式历史记忆检索 + 当前状态融合”

最终得到的是：
- baseline 1: 原始 pi05
- baseline 2: pi05_memory（显式 history memory）
- 后续 methods: 在 pi05_memory 上再加 future planning

==================================================
一、当前代码结构假设
==================================================

当前你已经有目录：

src/lerobot/policies/pi05_memory/

里面至少有这些文件：
- configuration_pi05_memory.py
- modeling_pi05_memory.py
- processor_pi05_memory.py

另外，你本地已经有 MemoryVLA 源码，Codex 可以查看，但这次只借鉴它“显式历史记忆”的思想，不要直接照搬整套代码。

==================================================
二、第一步：修改 configuration_pi05_memory.py
==================================================

请在 `Pi05MemoryConfig` 里新增以下字段：

use_history_memory: bool = True
memory_size: int = 16
memory_dim: int = 512
memory_num_heads: int = 8
memory_dropout: float = 0.1
memory_fusion: str = "gated"
reset_memory_on_new_episode: bool = True
use_time_embedding_in_memory: bool = True

推荐默认值：
- memory_size = 16
- memory_dim = 512
- memory_num_heads = 8
- memory_fusion = "gated"

如果当前 config 有 hidden_size，可以让 memory_dim 默认等于 hidden_size；如果不好处理，就先固定 512。

==================================================
三、第二步：在 modeling_pi05_memory.py 中新增三个模块
==================================================

请新增下面三个类。

------------------------------------------
1. HistoryMemoryEncoder
------------------------------------------

作用：
把当前 trunk 的状态摘要压缩成一个 memory token。

实现逻辑：
- 输入: [B, D]
- 输出: [B, Dm]

可按下面思路实现：

class HistoryMemoryEncoder(nn.Module):
    def __init__(self, input_dim, memory_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, memory_dim)
        self.norm = nn.LayerNorm(memory_dim)

    def forward(self, x):
        return self.norm(self.proj(x))

------------------------------------------
2. HistoryRetriever
------------------------------------------

作用：
用当前 query 去读取历史 memory tokens。

实现逻辑：
- query: [B, 1, Dq]
- memory_tokens: [B, T, Dm]
- 输出:
  - history_summary: [B, 1, Dm]
  - attn_weights

建议使用 nn.MultiheadAttention(batch_first=True)

可按下面思路实现：

class HistoryRetriever(nn.Module):
    def __init__(self, query_dim, memory_dim, num_heads, dropout):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, memory_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(memory_dim)

    def forward(self, query, memory_tokens, memory_mask=None):
        q = self.query_proj(query)
        out, attn = self.attn(
            q,
            memory_tokens,
            memory_tokens,
            key_padding_mask=memory_mask,
            need_weights=True,
        )
        return self.norm(out), attn

------------------------------------------
3. GatedMemoryFusion
------------------------------------------

作用：
把当前状态和历史摘要融合。

实现逻辑：
- current_state: [B, D]
- history_summary: [B, Dm]
- 输出 fused_state: [B, D]

建议先把 history_summary 投影到 hidden_dim，再做 gate。

可按下面思路实现：

class GatedMemoryFusion(nn.Module):
    def __init__(self, hidden_dim, memory_dim):
        super().__init__()
        self.memory_to_hidden = nn.Linear(memory_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, current_state, history_summary):
        h = self.memory_to_hidden(history_summary)
        g = torch.sigmoid(self.gate(torch.cat([current_state, h], dim=-1)))
        fused = g * h + (1.0 - g) * current_state
        return self.norm(fused)

==================================================
四、第三步：在 Pi05MemoryModel 的 __init__ 中注册这些模块
==================================================

请在主模型初始化时，增加：

1. self.history_memory_encoder
2. self.history_retriever
3. self.gated_memory_fusion

同时，如果 use_time_embedding_in_memory=True，可以增加一个简单的时间位置嵌入：
- self.memory_time_embedding = nn.Embedding(memory_size, memory_dim)

如果你觉得第一版先不加时间嵌入更稳，也可以先留接口，不在第一版里启用。

==================================================
五、第四步：找到 forward 中“进入 action head 之前”的状态，并插入 history memory
==================================================

这是整个改动里最关键的地方。

目标不是在视觉 backbone 早期加 memory，而是在：
“多模态 trunk 已经融合好当前状态，马上要送入 flow-matching action head”
这个位置加 history memory。

请让 Codex 在 modeling_pi05_memory.py 中找到大概这样的逻辑：

current_state = trunk(...)
action_output = action_head(current_state, ...)

然后修改成：

1. 从 current_state 里构造一个 state_summary
2. 用 state_summary 编码成当前 memory token
3. 从历史 memory tokens 中检索 history_summary
4. 用 current_state 和 history_summary 融合，得到 fused_state
5. 再把 fused_state 送入原来的 action head

------------------------------------------
state_summary 的实现建议
------------------------------------------

如果 current_state 是 token 序列 [B, N, D]：
先做 mean pooling：

state_summary = current_state.mean(dim=1)

如果 current_state 已经是 [B, D]：
直接使用即可。

第一版不要做复杂 global token 逻辑，mean pooling 就行。

------------------------------------------
具体 forward 逻辑建议
------------------------------------------

请按下面的语义顺序插入：

A. 先计算 trunk 输出
B. 从 trunk 输出得到 state_summary
C. 编码当前 memory token
D. 如果有历史 memory，就检索 history_summary
E. 如果没有历史 memory，就用零向量或跳过融合
F. 得到 fused_state
G. fused_state 送入原 action head

==================================================
六、第五步：训练阶段如何处理历史
==================================================

训练阶段，第一版不要搞复杂在线缓存，而是做“batch 内构造历史窗口”。

目标：
对当前样本 t，构造过去 memory_size 步的历史输入。

优先级如下：

方案 A（首选）：
如果当前 dataset / batch 已经包含 episode_id 和 frame_id，
那就根据 episode 内时间顺序，在 collate 或 forward 前构造：
- history tokens
- history mask

方案 B（更容易先跑通）：
如果当前 pipeline 不方便在线构造历史索引，
那就在 dataset wrapper 或 collator 里直接返回：
- batch["history_obs"]
- batch["history_mask"]

然后在 forward 中先把 history_obs 编码成 history_tokens。

第一版目标不是最优，而是跑通。

------------------------------------------
训练阶段建议新增的 batch 字段
------------------------------------------

history_obs
history_mask

可选：
history_state
history_frame_ids
history_episode_ids

==================================================
七、第六步：推理阶段加入 runtime FIFO memory buffer
==================================================

训练时可以依赖 batch 历史，推理时必须做真正在线 memory。

请在 policy / model 对象中加入运行时状态，例如：

self.runtime_memory_tokens = []
self.runtime_episode_id = None

act() 时的逻辑：

1. 如果当前是新 episode，并且 reset_memory_on_new_episode=True：
   清空 runtime_memory_tokens

2. 当前 observation 经过 trunk 后，得到 state_summary

3. 用 runtime_memory_tokens 作为历史 memory 做检索
   - 如果为空，则 history_summary = 0 或跳过融合

4. 生成 fused_state -> action chunk

5. 再把当前时刻的 memory_token append 到 runtime_memory_tokens

6. 如果长度超过 memory_size，则弹出最早一个

注意：
不要在生成当前动作之前，就把当前 memory token 放进去，否则会把“当前”也当作历史。

==================================================
八、第七步：如果你想参考 MemoryVLA 源码，Codex 应重点看什么
==================================================

请让 Codex 只参考这些概念：
1. 显式 memory token 序列
2. 当前 query 对历史做 attention retrieval
3. 当前状态与 history summary 的融合
4. 历史需要有时间顺序感

不要直接迁移这些：
1. OpenVLA 主干
2. perceptual/cognitive 双流复杂结构
3. consolidation / token merge
4. 它自己的训练脚本
5. 完整 planner 结构

原因：
你的骨架是 LeRobot pi0.5，不是 OpenVLA。

==================================================
九、第八步：先做最小可运行版本，不要加的内容
==================================================

第一版明确不要做：

- ordered latent planner
- future plan KV
- failure handling
- utility critic
- subtask prediction
- 双流 memory（perceptual / cognitive 分开）
- memory token merge
- 复杂跨时刻 consistency loss

只做：
- 历史显式存储
- 当前 query 检索历史
- gated fusion
- 原有 FM action head 不变

==================================================
十、第九步：启动脚本参数
==================================================

请在你当前的 .sh 启动文件中新增这些参数：

--policy.type=pi05_memory
--policy.use_history_memory=true
--policy.memory_size=16
--policy.memory_dim=512
--policy.memory_num_heads=8
--policy.memory_dropout=0.1
--policy.memory_fusion=gated
--policy.reset_memory_on_new_episode=true
--policy.use_time_embedding_in_memory=true

如果当前 CLI 参数注册方式不同，请按 LeRobot 当前方式适配。

==================================================
十一、给 Codex 的最终执行要求
==================================================

请让 Codex 最终输出这些内容：

1. 修改了哪些文件
2. 每个新增类的完整代码
3. forward 里 memory 插入的准确位置说明
4. 训练时历史窗口是怎么构造的
5. 推理时 runtime memory buffer 是怎么实现的
6. 如果 processor/collator 需要改，也要说明
7. 给出一个最小 smoke test 的运行命令

==================================================
十二、最终目标
==================================================

这次改完之后，你应该得到一个“仅补历史”的 baseline：

- 原始 pi05：无显式长程历史
- pi05_memory：有显式 history retrieval
- 后续你的方法：在 pi05_memory 上继续加 future planning

这就是当前阶段最合理、最稳的第二个 baseline。