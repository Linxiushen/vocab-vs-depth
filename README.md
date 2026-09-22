# vocab-vs-depth

**同样的参数预算，是花在更大的词表上、还是花在多一层 Transformer 上？** 这个仓库用一个
参数对齐、可复现、带统计检验的受控实验，正面回答了这个问题——并且把「怎样把这种小消融
做对」的全套方法学（字节级 BPB、嵌套 tokenizer、无 padding 打包、配对 bootstrap、
一整节诚实边界）开源出来。

基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）从零预训练。

[![license](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![weights](https://img.shields.io/badge/weights-CC--BY--NC--4.0-lightgrey)](#模型权重)
[![HF armA](https://img.shields.io/badge/🤗-armA%20v6400%20L8-yellow)](https://huggingface.co/Linxiushen/vocab-vs-depth-armA-v6400-L8)
[![HF armB](https://img.shields.io/badge/🤗-armB%20v16384%20L7-yellow)](https://huggingface.co/Linxiushen/vocab-vs-depth-armB-v16384-L7)

> **状态（2026-09-22）**：单种子 pilot 主对照实验（T=120M token，两臂各约 64M 参数）**已完成、已评测、已统计检验**，结论见下。
> 权重与各自的 tokenizer 发布在 Hugging Face（CC-BY-NC-4.0）。
> 多种子 / 多预算档的正式实验（需要租卡）是下一步。

---

## 结果速览（TL;DR）

在**同一份中文验证集、同一份 v3 评测切分、同一 token 预算**下，参数对齐的两臂：

| 臂 | 配置 | 参数量 | **BPB（越低越好）** |
|---|---|---:|---:|
| A | 词表 6400 · **8 层** | 63.91M | 0.858790 |
| **B** | **词表 16384** · 7 层 | 64.21M | **0.852450** |

**臂 B（大词表、少一层）胜出**，BPB 低 **0.006340**。配对 bootstrap（10,000 次重采样、
两臂共用重采样索引、19,866 篇文档 / 19,202,025 字节）：

- 95% 置信区间 **[0.005959, 0.006716]，不含 0**
- 效应量是自举标准误的 **≈ 32 倍**（stderr = 0.000195）
- 相对幅度 0.744%

也就是说：在这个规模和预算下，「把参数花在词表上」略优于「花在深度上」，而且这个差异
**远大于测量噪声**——不是抖出来的。

> ⚠️ 一句话边界：这是 **单种子 pilot**，只覆盖一份验证集的测量噪声，**不覆盖种子间训练方差**；
> 只测了 T=120M（≈1.9 token/参数，欠训练区间）一个预算点。完整边界见 [诚实边界](#诚实边界)。
> 结论应读作「**词表 6400 / 8 层 这一配置 vs 词表 16384 / 7 层 这一配置**，在此规模此预算下」。

---

## 为什么这个问题值得做

- **它是每个从零训练小模型的人都要做的真实取舍。** embedding 在小模型里占比很大：本项目里
  词表从 6400 扩到 16384，仅嵌入参数就多 7.67M——几乎正好等于加一层 Transformer 的 7.37M。
  预算固定时二者只能选一个，但很少有人用对齐实验量过。
- **大词表的收益容易被评测协议悄悄放大或抹掉。** 用 PPL 跨词表比较是错的；按篇 padding、
  或用某一臂的分词去切公共块，都会系统性偏向一方。本项目把这些坑一个个挖出来、落盘、绕开
  （见 [镜像混淆](#3-旧-v2-评测协议自带一个镜像混淆) 与 [v3 打包协议](#公平对比协议各臂独立-packing)）。
- **它是一个"把小消融做对"的完整范本。** 参数与 FLOPs 对齐、嵌套 tokenizer 消除 BPE 随机性、
  字节级 BPB、配对 bootstrap 预注册灵敏度、containment 近重复审计、以及一整节写在明处的
  局限——整套方法学可直接复用到别的消融上。

---

## 实验臂

「加一层」和「扩词表」的参数成本几乎相等，这让对比能干净对齐：

```
加 1 层 (d=768):        +7.37M
词表 6400 → 16384:      +7.67M
```

| 臂 | tokenizer | vocab | 层数 | hidden | FFN | 参数量 |
|---|---|---:|---:|---:|---:|---:|
| **A** | 本次 BPE 合并序列的 6.4K 前缀 | 6,400 | 8 | 768 | 2432 | 63.91M |
| **B** | 本次训练的 16K BPE | 16,384 | 7 | 768 | 2432 | 64.21M |

参数差 0.46%，输入 embedding 与输出投影共享权重，每 position 名义 FLOPs 差 0.71%。

**两个 tokenizer 是嵌套的**：6.4K 的 6108 条 merge 是 16K 的 16092 条的精确前缀，
A 的词表 ⊆ B 的词表且 token id 完全一致。这消除了「两次独立 BPE 训练的随机差异」这个混淆
（内部效度的优势），但也是外部效度的限制（见诚实边界）。

pilot 主实验：两臂各 T=120M token、7,339 次参数更新、seed 42、fp32、Apple Silicon MPS，
墙钟分别 17.4h / 16.4h。同一 `packed_doc_order_sha256`，即两臂看到的文档顺序完全相同，
只有分词与层数不同。臂 A 用 4,795,627 token、臂 B 用 4,052,414 token 覆盖同样的 19,202,025 字节。

---

## 方法学要点

### 跨词表必须用 BPB，不能用 loss/PPL

```
BPB = 总正文 NLL(nats) / (ln2 × 总原文 UTF-8 字节数)
```

词表越大、token 越少、每 token 承载信息越多、loss 天然越高——本次 per-token PPL 是
A=10.84 / B=16.44，**方向与 BPB 相反**。字节是与 tokenizer 无关的物理量，是唯一可跨臂比较的口径。

### 公平对比协议：各臂独立 packing

两臂各自把语料打包成**无 padding** 的连续 token 流按 512 切分。主轴等算力（等步数 ≈ 等 position），
副轴等原文字节。被否决的方案记录在 [lab/决策-20260918.md](lab/决策-20260918.md)，其中最关键一条：
「用某一臂的分词去切公共块」会把另一臂唯一的机制优势烧成 padding，等于把被测变量当混淆变量抹掉。

### 配对 bootstrap 的灵敏度

用逐文档 NLL 做配对重采样（两臂共用同一组重采样索引，抵消文档难度）。本次在 19,866 篇验证集上
达到的自举标准误是 0.000195——实测效应 0.006340 是它的约 32 倍。**这只覆盖测量噪声**；
seed 间训练方差是另一个、通常更大的不确定性来源，需要重复训练才能估计。见 `lab/power_analysis.py`。

### 学习率

上游 `get_lr` 在 t=0 就是满 lr、无 warmup。本项目补丁加入 WSD 日程（warmup 2% → stable →
cosine decay 10%）。两臂同用 lr=2e-4，**未按臂调 LR**（列为局限）。

### 3. 旧 v2 评测协议自带一个镜像混淆

按篇 padding 的 v2 协议下，小词表臂被迫多切上下文块（集中在长文上），方向上有利于大词表臂。
这正是本项目改用 v3 公共块协议的原因。落盘证据：`lab/results/v2-mirror-confusion.json`。

---

## 模型权重

两臂权重 + 各自的 tokenizer 发布在 Hugging Face，**CC-BY-NC-4.0，仅限非商业**：

| 臂 | Hugging Face |
|---|---|
| A（vocab 6400 / 8 层） | [Linxiushen/vocab-vs-depth-armA-v6400-L8](https://huggingface.co/Linxiushen/vocab-vs-depth-armA-v6400-L8) |
| B（vocab 16384 / 7 层） | [Linxiushen/vocab-vs-depth-armB-v16384-L7](https://huggingface.co/Linxiushen/vocab-vs-depth-armB-v16384-L7) |

这是**受控实验的两个臂**，不是可用的聊天模型：只做过预训练、没有任何指令微调或对齐、规模很小。
加载时**必须用对应仓库自带的 tokenizer**（两臂分词器不同，换错结果没有意义）。每个仓库的
模型卡都写明了配置、结果、边界与加载方式。

---

## 复现

```bash
git clone --recurse-submodules https://github.com/Linxiushen/vocab-vs-depth.git
cd vocab-vs-depth
./lab/apply_upstream_patch.sh          # 幂等，把本项目补丁打到 pin 住的上游

# 语料不在本仓库内，自行从上游获取（见 DATA_LICENSE.md 判断你的用途是否合规）
#   ModelScope: gongjy/minimind_dataset -> pretrain_t2t_mini.jsonl

.venv/bin/python lab/pack_corpus.py --arm A   # 各臂独立无 padding 打包
.venv/bin/python lab/pack_corpus.py --arm B

# 训练（pilot 口径：T=120M token，seed 42）
.venv/bin/python lab/run_pretrain.py --arm A --packed lab/data/v3/packed_A \
    --total_tokens 120e6 --out lab/runs/pilot-A-s42-T120M
.venv/bin/python lab/run_pretrain.py --arm B --packed lab/data/v3/packed_B \
    --total_tokens 120e6 --out lab/runs/pilot-B-s42-T120M

# v3 评测（臂 B 示例；臂 A 换 tokenizer / vocab_size / num_hidden_layers 与权重）
.venv/bin/python lab/eval_bpb.py --protocol v3 \
    --weight lab/runs/pilot-B-s42-T120M/weights/pretrain_768_step0007339.pth \
    --tokenizer lab/tokenizers/bpe_16384 --vocab_size 16384 --num_hidden_layers 7 \
    --val lab/data/v3/val_large.jsonl --chunks lab/data/v3/eval_chunks_v3.jsonl \
    --output lab/results/armB-T120M-eval-v3.json

# 配对 bootstrap
.venv/bin/python lab/power_analysis.py \
    --a lab/results/armA-T120M-eval-v3.json --b lab/results/armB-T120M-eval-v3.json \
    --resamples 10000 --out lab/results/paired-armA-vs-armB.json
```

完整结果说明见 [lab/结果-两臂主实验.md](lab/结果-两臂主实验.md)。

---

## 诚实边界

这一节是项目的一部分，不是免责声明。

1. **单种子。** 每臂只训练了 seed 42 一个种子，所以 95% CI 只覆盖**这份固定验证集上的测量噪声**，
   **不能排除**换个种子结论反转。要支撑普遍性结论必须每臂多次重复训练——这是正式实验的核心目标。
2. **这是打包对比，不是纯词表消融。** 两臂同时差三样：词表 6400 vs 16384、层数 8 vs 7、
   tied embedding 4.9M vs 12.6M。正确表述是「6400词表8层 这一配置 vs 16384词表7层 这一配置」。
3. **预算条件性 / 欠训练。** T=120M token ≈ 1.9 token/参数，远低于 Chinchilla 的 20。
   所有结论限定在欠训练区间，不能外推到充分训练或更大模型。
4. **大词表在随机初始化时就有 BPB 结构性先手**（`ln(V)/bytes_per_token` 项，与学习无关）。
   本项目实测臂 A 随机初始化 BPB = 3.21（`lab/results/armA-random-eval-v3.json`），未单独跑臂 B
   的随机基线，因此不量化 B 的先手大小。**不能把 B 的训练后微弱领先解读为「大词表学得更好」**——
   方向相反的两种力同时存在。
5. **两个 tokenizer 嵌套**：内部效度的优势、外部效度的限制。独立训练的 6400 词表会得到不同的
   merge（官方 MiniMind 6.4K 是 1.4872 字符/token，本项目同源 6.4K 是 1.6061）。
6. **只测 BPB。** 无下游任务、无生成质量、无推理时延与显存对比。只能说「B 的 BPB 更低」，
   不能说「B 的模型更好」。
7. **packing 引入训练/评测失配**（对两臂近似对称）：训练序列可能从文档中段开始、有跨文档上下文；
   评测每块从 BOS 开始且不跨文档。未实现跨文档 block-diagonal mask，不宣称消除了跨文档注意力污染。
8. **单一语料、单一 seq_len 512、单一 lr、单一架构。**

数据隔离（按内容哈希划分、containment 近重复审计、只发哈希清单不发原文）与三条已更正的旧说法，
见仓库历史与 [DATA_LICENSE.md](DATA_LICENSE.md)、[lab/结果-两臂主实验.md](lab/结果-两臂主实验.md)。

---

## 学习路径

1. [第一课：一次预训练到底发生了什么](lab/第一课-一次预训练到底发生了什么.md)
2. 用 Archify 制作的[项目全流程图](lab/diagrams/training-map.html) 与[单批训练循环图](lab/diagrams/training-step.html)，配[逐步讲解](lab/diagrams/流程图讲解.md)
3. [课题设计](lab/课题设计.md) 与 [决策记录](lab/决策-20260918.md)
4. 看模型续写：`.venv/bin/python lab/sample_pretrain.py --prompt "学习语言模型需要"`

---

## 许可

- **代码**：Apache-2.0（[LICENSE](LICENSE) / [NOTICE](NOTICE)）。
- **第三方素材**：[THIRD_PARTY.md](THIRD_PARTY.md)。
- **训练数据**：**不在本仓库分发**，上游带非商业条款，只提供内容哈希清单与重建脚本（[DATA_LICENSE.md](DATA_LICENSE.md)）。
- **模型权重**：CC-BY-NC-4.0，**不得用于任何商业产品**（包括作者自己的）。

上游 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) pin 在
`a3c7b01cc004d5de86aea961f20bf1e638e7c09e`，以 git submodule 引用；本项目对上游的唯一改动是
`lab/upstream-pretrain.patch`。这不是法律意见。
