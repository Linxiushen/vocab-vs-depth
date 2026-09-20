# vocab-vs-depth

在约 64M 参数预算下，**把参数花在更大的词表上、还是花在多一层 Transformer 上**，
哪个得到的中文模型 BPB 更低？

这是一个学习性质的从零预训练项目，基于
[jingyaogong/minimind](https://github.com/jingyaogong/minimind)（Apache-2.0）。
目标是**理解预训练的每一个环节**，不是产出一个可用的聊天模型。

> 更新：2026-09-19。**64M 正式对照实验尚未开始**（需要租卡）。
> 本仓库当前提供的是：一次已完成并独立验证的 30M 预训练、两个自训 tokenizer、
> 以及为 64M 实验准备好的完整数据/打包/评测/训练链路。

---

## 研究问题与实验臂

实测结果显示，**加一层和扩词表的参数成本几乎相等**，这让实验能干净对齐：

```
加 1 层 (d=768):        +7.37M
词表 6400 → 16384:      +7.67M
```

| 臂 | tokenizer | vocab | 层数 | hidden | FFN | 参数量 |
|---|---|---:|---:|---:|---:|---:|
| **A** | 本次 BPE 合并序列的 6.4K 前缀 | 6,400 | 8 | 768 | 2432 | 63.91M |
| **B** | 本次训练的 16K BPE | 16,384 | 7 | 768 | 2432 | 64.21M |

参数差 0.46%，是近似匹配。输入 embedding 与输出投影共享权重。
每 position 名义 FLOPs：A = 421,221,888，B = 418,262,400（差 **0.71%**）。

**两个 tokenizer 是嵌套的**：6.4K 的 6108 条 merge 是 16K 的 16092 条的精确前缀，
A 的词表 ⊆ B 的词表且 token id 完全一致。这对内部效度是优势（消除了"两次独立
BPE 训练的随机差异"），对外部效度是限制（见下方诚实边界）。

---

## 目前的可信结果

### 1. 30M 模型确实学到了东西

同一 tokenizer、同一批完整原文、同一评分协议（`all_body_tokens_bos_per_chunk_v2`）：

| 模型 | 验证 token loss | BPB（越低越好） |
|---|---:|---:|
| 随机初始化 30M | 8.8374 | 3.4230 |
| 已预训练 30M | 4.6974 | 1.8194 |

BPB 降低 **46.8%**。这衡量的是对未见文本的预测能力，**不代表**对话或知识任务准确率。

模型规格 30,025,216 参数、40,000 篇文本、5,000 micro-batch / 1,250 次参数更新、
**8,141,013 个有效监督 token**（含 EOS），padding 占 60.1%。

### 2. 大词表显著减少了文本 token

在相同 1,943 篇文档、784,682 个字符上完整编码，编解码全部无损：

| tokenizer | token 总数 | 字符/token |
|---|---:|---:|
| 官方 MiniMind 6.4K（参考） | 527,624 | 1.4872 |
| 本次同源 6.4K（对照） | 488,574 | 1.6061 |
| 本次 16K | 413,143 | **1.8993** |

同源 16K 比同源 6.4K 少 **15.4%** 的 token。16K 用 30 万篇 / 287,727,252 字节
训练文本学习，BPE 耗时 626.6 秒；6.4K 取同一学习过程的前缀合并序列，
**不是另一次独立训练**。

这是压缩结果，**还不能说明少一层的模型质量更好，也不等于训练加速 15.4%**。

### 3. 现有评测协议 v2 自带一个镜像混淆

这是本项目为什么要引入 v3 评测协议的原因。在 1,943 篇验证集上按 510 token 切块：

| | A（6.4K） | B（16K） |
|---|---:|---:|
| 需要的块数 | 2,169 | 2,087 |

A 被迫多切 82 块 = 多 **3.929%** 次上下文重置，集中在 **143 篇长文**上
（占验证集 29.5% 的字节）。方向上有利于 B。
落盘证据：`lab/results/v2-mirror-confusion.json`

---

## 实验设计（v3）

### 公平对比协议：各臂独立 packing

两臂各自把语料打包成**无 padding** 的连续 token 流，按 512 切分。
主轴是**等算力**（等步数 ≈ 等 position ≈ 等墙钟，实测两臂 ms/step 差 1.1%），
副轴是**等原文字节**（B 独立完整退火的 run，不用中途 checkpoint）。

被否决的方案及理由记录在 [lab/决策-20260918.md](lab/决策-20260918.md)，其中最重要的一条：
「A 驱动的共同文本分块」会让 B 每块只有 433 个真 token、79 个 padding，
**等于把 B 唯一的机制优势烧成 padding**，把被测变量当混淆变量抹掉。

### 学习率

上游 `get_lr` 是 `lr*(0.1+0.45*(1+cos(pi*t/T)))`，**t=0 时就是满 lr，完全没有 warmup**。
本项目补丁加入 WSD 日程（warmup 2% → stable → cosine decay 10%），
上游 cosine 行为保留不变。两臂同用 lr=2e-4，**未按臂调 LR**（列为局限）。

### 评测

`lab/eval_bpb.py` 支持两个协议，v2 保持向后兼容（有逐字节回归基准）：

```
BPB = 总正文 NLL(nats) / (ln2 × 总原文 UTF-8 字节数)
```

跨词表比较**必须用 BPB 不能用 loss/PPL**：词表越大 token 越少、每 token 承载信息越多、
loss 天然越高。字节是与 tokenizer 无关的物理量。

### 预注册的灵敏度

用 1,943 篇的逐文档 NLL 做配对 bootstrap 实测：
**能可靠检出的最小 BPB 差是 0.024 绝对值 / 1.32% 相对值**（`lab/power_analysis.py`）。

这只覆盖**测量噪声**。seed 间的训练方差是另一个、通常更大的不确定性来源，
需要重复训练才能估计。若最终 |Δ| 小于达成的 MDE，只能写"未能区分"，
**不得写"无差异"或"等价"**。

---

## 数据与隔离

原始语料 1,270,238 行，派生自 `jingyaogong/minimind_dataset` 的 `pretrain_t2t_mini.jsonl`。

冻结数据在 `lab/data/v3/`（带 SHA-256 与重叠审计）：

| 文件 | 行数 | 说明 |
|---|---:|---|
| `train.jsonl` | 1,250,235 | 训练源文档 |
| `val_large.jsonl` | 19,866 | 固定验证集，前 1,924 篇是存活的 v2 历史篇目 |
| `val_manifest.jsonl` | 19,866 | 内容 SHA-256 清单（**仓库里只发这个，不发原文**） |
| `eval_chunks_v3.jsonl` | 21,915 | v3 共同文本分块 |

划分按**文本内容哈希**，不是按文件位置——因为实测发现语料**按来源排序、完全没有 shuffle**
（头部是中文指令/诗歌，中位 231 字符；尾部是英文考试选择题，中位 1,251 字符）。
早期从尾部切验证集，切出来的全是英文考题。

### 近重复审计

`lab/neardup_audit.py` 用 **containment 覆盖率**判据，不用 Jaccard
（Jaccard≥0.8 查不出包含式泄漏——短的 val 文档被抄进长 train 文档的某一段）。

参数：window=64 / stride=32 / min_hits=3 / min_coverage=0.5 / max_doc_freq=100。
结果：剔除 **134 篇**（命中率 0.67%），其中原 1,943 篇里剔了 **19 篇**。

**为什么不用朴素的"共享窗口计数"判据**：实测它会被文档开头的公共系统提示词触发，
且剔除是按语种定向的——被剔集 52.30% 是 >80% ASCII 的英文文档，保留集只有 8.58%（**6.1 倍**）。
本课题的因变量是 BPB（每字节比特数），自变量是词表大小，**定向删英文会改变评测集的
字节构成，而字节构成恰好与词表压缩率交互**，那会让结论变成协议选择的产物。
换成覆盖率 + 文档频次过滤后，偏置从 6.10× 降到 **0.40×**，剔除量只占总字节 0.63%。

敏感性分析（两种判据、多个阈值）落盘在 `lab/data/v3/neardup_report.json`。

---

## 复现

```bash
git clone --recurse-submodules <this repo>
cd vocab-vs-depth
./lab/apply_upstream_patch.sh          # 幂等，把本项目补丁打到 pin 住的上游

# 语料不在本仓库内，自行从上游获取（见 DATA_LICENSE.md 判断你的用途是否合规）
#   ModelScope: gongjy/minimind_dataset -> pretrain_t2t_mini.jsonl
#   sha256 = 6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c

.venv/bin/python scripts/rebuild_val.py      # 按哈希清单还原验证集，校验 sha256
.venv/bin/python lab/pack_corpus.py --arm A  # 打包，四条硬断言
.venv/bin/python lab/pack_corpus.py --arm B
.venv/bin/python lab/run_pretrain.py --arm A --packed lab/data/v3/packed_A \
    --total_tokens 240e6 --batch_size 64 --out lab/runs/main-A-s42-T240M
.venv/bin/python lab/eval_bpb.py --protocol v3 --weight <ckpt> --tokenizer <arm tokenizer> \
    --val lab/data/v3/val_large.jsonl --chunks lab/data/v3/eval_chunks_v3.jsonl
.venv/bin/python -m unittest discover -s lab -p 'test_*.py'
```

`lab/bench_m5.py`（Apple Silicon）与 `lab/bench_gpu.py`（CUDA）用于在正式开跑前
标定真实吞吐——**所有工期估算都必须用实测值，不用外推值**。

### 算力现实

| | 实测 |
|---|---|
| MacBook Air M5 24GB，臂 A 64M seq512 batch8 | 1087 ms/step，3,768 tok/s |
| 同上，臂 B | 1075 ms/step，3,812 tok/s |
| M5 跑满 1 epoch | 单臂约 48 小时，两臂约 95 小时 —— **不可行** |

正式实验计划租 RTX 4090，2 臂 × 2 个预算档 × 多 seed，机时 20–36 小时。
M5 只做数据准备与冒烟。

---

## 诚实边界

这一节是项目的一部分，不是免责声明。

1. **这是打包对比，不是词表消融。** 两臂同时差三样：词表 6400 vs 16384、层数 8 vs 7、
   tied embedding 4.915M vs 12.583M。正确表述是"6400词表8层 这一配置 vs 16384词表7层 这一配置"。
2. **预算条件性。** 计划的主档只有约 3.76 token/参数，Chinchilla 比例是 20。
   所有结论限定在欠训练区间，不能外推到充分训练或更大模型。
3. **不能宣称"欠训练不利于大词表臂，所以 B 赢是保守下界"。** 随机初始化的解析 BPB 是
   A ≈ 3.16、B ≈ 2.95 —— **B 在零训练时就先赢 0.205 BPB**（因为 `ln(V)/bytes_per_token`）。
   方向相反的两种力同时存在，量级都不明。
4. **两个 tokenizer 嵌套** 是内部效度的优势、外部效度的限制。独立训练的 6400 词表会得到
   不同的 merge——本项目自己的数据就是证据：官方 MiniMind 6.4K 是 1.4872 字符/token，
   同源 6.4K 是 1.6061。
5. **只测 BPB。** 无下游任务、无生成质量、无推理时延与显存对比。
   不能说"B 的模型更好"，只能说"B 的 BPB 更低/更高"。
6. **packing 引入训练/评测失配**（对两臂近似对称但影响绝对数值）：训练序列可能从文档中段
   开始、有跨文档上下文；评测每块从 BOS 开始且不跨文档；RoPE 位置不在文档起点重置；
   训练对 BOS/EOS 计损失（占目标 A 1.26% / B 1.49%），评测不计。
7. **未实现跨文档 block-diagonal mask**（上游 attention 只接受 2D key-mask，
   传非全 1 会关掉 flash SDPA）。不能宣称消除了跨文档注意力污染。
8. **近重复审计用 64 位 blake2b**，理论上存在哈希碰撞。已剔除的 134 篇里人工复核过样本，
   确认是真实逐字重合而非碰撞误报，但未做全量二次校验。
9. **"我只用了 pretrain 文件所以没碰到 ShareAlike" 无法被证明。** 上游 pretrain 文件没有
   逐文件来源标注，其来源表述是"包括但不限于"。详见 DATA_LICENSE.md。
10. **验证集与 30M 历史结果的可比性已部分损失**：近重复审计剔除了原 1,943 篇里的 19 篇，
    所以"逐篇可比"现在只对存活的 1,924 篇成立。
11. **单一语料、单一 seq_len 512、单一 lr、单一架构。**

### 三个必须更正的旧说法

项目早期文档里这三条是错的，在此更正：

- **"臂 B 在 510 token 窗口下多读 18% 原文"** —— 在**按篇截断 + padding** 的协议下是错的，
  实测只有 **+2.5%**（93–95% 的文档短于 510 token，两臂都完整读完）。
  +18.4% 只在 packing 之后的等步数口径下成立。
- **每 position 名义 FLOPs 匹配到 0.71%**，不是 0.46%（后者只算了核心项，漏了注意力项）。
- **step-0 loss ≈ ln(V) 有系统性偏移。** 在 hidden=768 / vocab=6400 下实测稳定在 **8.90**，
  比 ln(6400)=8.7641 高约 0.14 nat；hidden=512 的 30M 模型偏高 0.073。
  已用 5 个 seed + 纯随机 token 独立复现，是模型架构与初始化的固有属性。
  所以"step-0 loss ≈ ln(V) ± 0.05"这类断言在 hidden=768 下不成立。

---

## 学习路径

1. [第一课：一次预训练到底发生了什么](lab/第一课-一次预训练到底发生了什么.md)
2 用 Archify 制作的[项目全流程图](lab/diagrams/training-map.html) 与
   [单批训练循环图](lab/diagrams/training-step.html)，配
   [逐步讲解与代码对照](lab/diagrams/流程图讲解.md)
3. [课题设计](lab/课题设计.md) 与 [总设计师决策](lab/决策-20260918.md)
4. 看自己模型的续写（有重复和跑题，见 `lab/results/smoke-samples.json`）：
   ```bash
   .venv/bin/python lab/sample_pretrain.py --prompt "学习语言模型需要"
   ```

---

## 许可

- **代码**：Apache-2.0。见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
- **第三方素材**：见 [THIRD_PARTY.md](THIRD_PARTY.md)（Archify MIT、JetBrains Mono OFL-1.1）。
- **训练数据**：**不在本仓库分发**。上游带非商业条款，只提供内容哈希清单与重建脚本。
  许可链、保守结论与三条残留不确定性见 [DATA_LICENSE.md](DATA_LICENSE.md)。
- **模型权重**：尚未发布。发布时将取 `cc-by-nc-4.0` 保守路线。
  **本项目的权重不得用于任何商业产品**，包括作者自己的产品。

这不是法律意见。

上游：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)，
pin 在 `a3c7b01cc004d5de86aea961f20bf1e638e7c09e`（2026-09-10），以 git submodule 引用。
本项目对上游的唯一改动是 `lab/upstream-pretrain.patch`。
