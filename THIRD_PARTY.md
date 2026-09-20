# 第三方素材与许可

本文件列出本仓库再分发的第三方素材及其许可义务。
最后核实：2026-09-18。

---

## Archify 2.17.0-dev.1 — MIT License

`lab/diagrams/training-map.html` 与 `lab/diagrams/training-step.html`
由 Archify 生成（https://github.com/tt-a1i/archify），文件内嵌其运行时代码。

**为什么 MIT 全文要抄在这里**：MIT 唯一的条件是"上述版权声明与本许可声明须包含
在软件的所有副本或实质性部分中"。实测这两个 HTML 内 `MIT License` 与 `tt-a1i`
均为 0 命中，即生成物没有自带许可文本。因此在此单独提供，以满足该条件。

```
MIT License

Copyright (c) 2026 tt-a1i (Archify)
Copyright (c) 2025 Cocoon AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## JetBrains Mono — SIL Open Font License 1.1

同样两个 HTML 内嵌了 JetBrains Mono 字体
（https://github.com/JetBrains/JetBrainsMono）。

与 Archify 的情况不同，**OFL-1.1 全文已经内联在每个 HTML 文件内部**
（实测命中，含 "SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007" 全文）。

⚠️ 再分发这两个 HTML 时**不得删除或修改**其中内联的 OFL 文本。

---

## 已移出本仓库的第三方代码

`lab/tools/archify/` 曾是 Archify 上游仓库的完整拷贝（pin 在 `a07fa1d`），
现已移出本仓库（本项目代码对它零引用）。若需重新生成流程图，请自行克隆
https://github.com/tt-a1i/archify。

---

## 上游 MiniMind

见根目录 `LICENSE`（Apache-2.0）与 `NOTICE`。
