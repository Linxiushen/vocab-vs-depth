---
license: cc-by-nc-4.0
language:
- zh
library_name: transformers
tags:
- minimind
- research
- ablation
---

# vocab-vs-depth 臂 B：vocab=16384, 7 层

这是一个**受控消融实验的一个臂**，不是一个通用可用的聊天模型。它只做过预训练，
没有做任何指令微调或对齐，规模也很小（120M 训练 token）。发布它的目的是让
[词表-深度权衡实验](https://github.com/Linxiushen/vocab-vs-depth) 的结果可复现。

## 实验设计

问题：在参数量大致对齐的前提下，小词表多一层和大词表少一层哪个更好？
两臂只差分词器与层数，训练数据的文档顺序完全相同。

| | 臂 A | 臂 B |
|---|---|---|
| vocab_size | 6400 | 16384 |
| num_hidden_layers | 8 | 7 |
| hidden_size | 768 | 768 |
| **本仓库** | 否 | **是**  |

## 结果

在同一份中文验证集（19,866 篇文档，19,202,025 UTF-8 字节）上，
用 bits-per-byte（bpb）评测：

| | 臂 A | 臂 B |
|---|---|---|
| **bpb**（越低越好） | 0.858790 | 0.852450 |

配对自举（10000 次重采样）：A − B = 0.006340 bpb，
95% CI [0.005959, 0.006716]，不含 0。**在这个 token 预算下臂 B 更好**
（本仓库是臂 B，这一臂更好）。

指标用 bpb 而不是 perplexity：两臂词表不同，token 数不可比
（A 用 4,795,627 token、B 用 4,052,414 token 覆盖同样的字节），
per-token 的 ppl 跨臂没有意义。

### 这个结论的边界

- 95% CI 只覆盖**这一份固定验证集上的测量噪声**。
- 每臂只训练了**一个种子**（42），所以**不能**排除换种子后结论反转。要支撑普遍性结论
  需要每臂多次重复训练。
- 只测了 T=120M token 这一个预算，没有测词表收益随预算变化的曲线。

## 怎么加载

权重是 `state_dict`，不是 `from_pretrained` 格式，需要配 minimind 的模型定义：

```python
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # 来自 minimind 仓库

tok = AutoTokenizer.from_pretrained("Linxiushen/vocab-vs-depth-armB-v16384-L7")
config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=7,
    vocab_size=16384,
    bos_token_id=tok.bos_token_id,
    eos_token_id=tok.eos_token_id,
)
model = MiniMindForCausalLM(config)
state = torch.load("pretrain_768_step0007339.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

分词器**必须**用本仓库自带的那一份：两臂的分词器不同，换错了结果没有意义。

## 训练

- 120,000,000 token，7339 个更新步，每步 16,352 token
- 随机初始化（不是从任何 checkpoint 继续），seed 42
- fp32，Apple Silicon MPS，墙钟 16.4 小时
- torch 2.14.0

## 校验和

- 权重 sha256：`72d55ddb25772711ef0932ee893b6b3360333e08b5067c099d599ab06bab8cf1`
- 分词器 sha256：`ff99beb3a8ea206db443a3247229b637c16ba6d2faa0caab71f36dc925addfb8`
- 验证集 sha256：`bac92439dfbad231c0381e72c42ef86d3d0cc073748b20c6ec0d460817e3eef1`
- 评测 chunk sha256：`51dc06f74152d687dbc527348d2a58c9e04c1222be5d8bca595a221795ea4ed0`

## 许可与出处

`cc-by-nc-4.0`，与上游训练语料的许可一致，仅限非商业使用。
模型代码来自 [minimind](https://github.com/jingyaogong/minimind)。
实验管线、评测协议与结果：https://github.com/Linxiushen/vocab-vs-depth

模型只经过预训练，没有做安全对齐；它会生成不可靠或不当的文本，不要直接用于任何面向用户的场景。

*本卡片与实验管线在 AI 辅助（Claude）下完成；数字均由上述脚本实跑产出。*
