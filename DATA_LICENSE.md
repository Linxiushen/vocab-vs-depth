# 训练数据的许可链与本项目的处理方式

> **这不是法律意见。** 本文件记录的是作者能核实到的事实，以及基于这些事实
> 采取的保守做法。若你要把本项目的任何产物用于商业用途，请自行咨询律师。

最后核实：**2026-09-18**（表中每个 URL 均在该日期由作者逐个访问确认）。

---

## 一、本项目实际用到的数据

只有一个文件：`pretrain_t2t_mini.jsonl`，来自 `jingyaogong/minimind_dataset`。
1,270,238 行，SHA-256 `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`。

本项目**未使用** SFT 数据（`sft_t2t_mini.jsonl` 等）。这一点后面很重要。

---

## 二、许可链（逐个核实）

| 环节 | 声明的许可 | 核实来源 |
|---|---|---|
| jingyaogong/minimind 代码 | Apache-2.0，**仓库内无 NOTICE 文件** | https://github.com/jingyaogong/minimind ；本地 `ls minimind \| grep -i notice` 为空 |
| minimind_dataset（HuggingFace） | YAML 里同时列出 `apache-2.0` 与 `cc-by-nc-2.0` | https://huggingface.co/datasets/jingyaogong/minimind_dataset |
| minimind_dataset（ModelScope） | **`CC-BY-NC-4.0`**（与 HF 声明不一致） | https://www.modelscope.cn/api/v1/datasets/gongjy/minimind_dataset |
| deepctrl-sft-data（匠数） | Apache-2.0 | https://www.modelscope.cn/api/v1/datasets/deepctrl/deepctrl-sft-data |
| Magpie-Align（上游只给了**组织名**，未指明具体数据集） | **混杂**：Magpie-Pro-300K-Filtered = `llama3`；Magpie-Qwen2-Pro-1M-v0.1 = YAML 无 license 字段 | https://huggingface.co/datasets/Magpie-Align |
| BAAI/COIG | `apache-2.0`，但卡内注明含 MIT 子集与 "fair use" 网爬数据 | https://huggingface.co/datasets/BAAI/COIG |
| ServiceNow-AI/R1-Distill-SFT | **`cc-by-nc-sa-4.0`**（NC **加** ShareAlike 传染） | https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT |
| stepfun-ai/Step-3.5-Flash-SFT | `apache-2.0` **且** `cc-by-nc-2.0`；卡内明写"必须同时遵守两个许可，不是二选一" | https://huggingface.co/datasets/stepfun-ai/Step-3.5-Flash-SFT |

上游对数据来源的原始表述见 `minimind/README.md`，注意它用的是
"**包括但不限于**……蒸馏补充语料"这样的开放式措辞。

---

## 三、从这些事实能推出什么

**1. 仓库级 NC 覆盖本项目用到的文件。**
无论取 HF 的 `cc-by-nc-2.0` 还是 ModelScope 的 `CC-BY-NC-4.0`，仓库级声明都覆盖
全部文件，**包括 `pretrain_t2t_mini.jsonl`**。所以本项目的语料受非商业条款约束。

**2. 带 ShareAlike 传染性的那一个，本项目没碰到。**
唯一带 SA 的是 `R1-Distill-SFT`（`cc-by-nc-sa-4.0`），它属于 **SFT** 数据链。
本项目只用了 pretrain 文件。这是本项目运气好的地方，但见下面第 (b) 条不确定性。

**3. CC BY-NC 无 ShareAlike，所以派生物可以换许可标签，但不能超出 NC 边界。**
CC BY-NC 2.0 与 4.0 的法律文本均无 ShareAlike 条款（已核实
https://creativecommons.org/licenses/by-nc/2.0/legalcode ）。NC 条款原文：
"You may not exercise any of the rights granted to You in Section 3 above in any
manner that is primarily intended for or directed toward commercial advantage or
private monetary compensation."

---

## 四、本项目的处理方式

### 语料：一律不再分发

本仓库**不包含任何训练或验证语料**，只提供：

- `lab/data/v3/val_manifest.jsonl` —— 每篇验证文档的**内容 SHA-256**，不含原文
- `scripts/rebuild_val.py` —— 第三方自行从上游下载 `pretrain_t2t_mini.jsonl` 后，
  按哈希查表按序还原出逐字节相同的验证集

这一个工程决定同时消除了四个问题：CC BY-NC §4(a)「不得对 Work 本身另发许可」
（原样摘录的他人文本，我无权给它换许可标签）；HF dataset 仓的 dataset card
披露义务；采样 RNG 流的复刻陷阱；以及 1.2GB 的仓库体积。

### 代码：Apache-2.0

见 `LICENSE` 与 `NOTICE`。代码是本项目原创或对上游 Apache-2.0 代码的补丁。

### 未来的模型权重与 tokenizer：保守取 `cc-by-nc-4.0`

训练产物是否继承训练数据的 NC 限制，在法律上**有争议**且无定论。
HF 上的同类实践并不统一——例如上游 `jingyaogong/minimind-3-pytorch` 用同一份
NC 数据训练却发布为 `apache-2.0`。

本项目取**保守路线**（`cc-by-nc-4.0`），理由是作者本人有商业硬件产品计划
（MicroCat 机器猫），一旦将来产生争议，激进的许可选择会直接波及产品。
宁可现在标严，也不要给未来埋雷。

> **本项目的权重不得用于任何商业产品，包括作者自己的产品。**
> 若要产品化，必须改用许可白名单语料重新训练。

tokenizer（BPE 合并表）与权重同等对待。虽然合并表只是从语料统计出的频次结构、
其"可版权性"比权重更可疑，但本项目不在这个问题上赌。

---

## 五、残留的不确定性（必须如实披露）

**(a) 上游两个平台的许可声明互相矛盾。**
HF 写 `[apache-2.0, cc-by-nc-2.0]`，ModelScope 写 `CC-BY-NC-4.0`。
本项目按"两者取严"处理，但这个矛盾本身未被上游澄清。

**(b) "我只用了 pretrain 文件所以没碰到 ShareAlike" 无法被证明。**
上游的 pretrain 文件**没有逐文件的来源标注**，其来源表述是"包括但不限于"。
因此无法排除 pretrain 文件里混有 SFT 链条中的内容。第三节第 2 条的结论
是基于上游文档的合理推断，不是可验证的事实。

**(c) 无法确认语料是否含 Llama 3 生成文本。**
上游只给了 `Magpie-Align` 的**组织名**，未指明用了哪个数据集。该组织下既有
Qwen2 系（宽松/未声明）也有 Llama-3 系（`llama3` 许可）。

Meta Llama 3 社区许可要求：用 Llama Materials 训练的模型必须显著展示
"Built with Meta Llama 3"，且**模型名称开头要带 "Llama 3"**
（https://www.llama.com/llama3/license/ ）。

本项目**未作此标注**，因为无法确认该要求是否适用。如果上游能澄清 Magpie 来源、
且确实含 Llama 3 生成文本，本项目的模型命名与标注需要相应修改。

---

## 六、如果你要复现本项目

1. 自行从上游获取 `pretrain_t2t_mini.jsonl`（ModelScope 或 HuggingFace），
   并自行判断你的用途是否符合其非商业条款。
2. 用 `scripts/rebuild_val.py` 还原验证集，核对 SHA-256。
3. 本仓库的代码按 Apache-2.0 授权，你可以自由使用——但你训练出的产物，
   其许可状态取决于你使用的数据，与本仓库的代码许可无关。
