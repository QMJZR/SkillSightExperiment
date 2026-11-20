# LLM 辅助标注方案（单人工版）

> 约束：只有一名人工标注者。目标不是用 AI 替代人工，而是用 4 个模型做预标和对照，把人工时间压到可完成的范围，同时保证论文里的标签仍然有人工验证。
> 制定日期：2026-11-20。

---

## 1. 角色分工

| 角色 | 做什么 | 不做什么 |
|---|---|---|
| 人工（你） | 校准 30 对；确认 pilot 全部 500 对；确认 validation 的全部 FM/FVA 候选和 20% 随机样本；7 天后自复查 100 对 | 不做无意义的重复劳动 |
| DeepSeek / Qwen / 智谱 / OpenAI | 独立预标同一批 pair；提供 evidence 和 confidence；多数票作为预标 | 不作为最终 ground truth；不参与方法设计 |
| 脚本 | 合并标签、算 LLM–human 一致性、抽样审计、生成 query 级主因 | 不替人做裁决 |

四个模型只用于**对照和预标**。最终标签以人工确认为准。

如果 OpenAI 网络不通，用 DeepSeek / Qwen / 智谱三家多数票（≥2/3），OpenAI 作为可选对照；门槛和审计规则不变。三家也凑不齐时，退化为“单模型预标 + 人工确认全部 FM/FVA”，论文里如实写。

## 2. 三个阶段

### 阶段 A：30 对校准

1. 你按 `02_标注手册.md` 标 30 对，得到人工标签。
2. 四个模型各跑一遍这 30 对，输入只有 query、参考答案、skill 名/描述/正文片段，不输入 gold、rank、retriever。
3. 算每个模型的：
   - semantic 加权 κ；
   - coverage 加权 κ；
   - FM 的 precision / recall / F1；
   - FVA 的 recall；
   - invalid_reason 的准确率。
4. 通过线：
   - 人工 7 天自复查 κ ≥ 0.70（这一项在阶段 C 才算）；
   - 四模型多数票对人工：FM F1 ≥ 0.75，FVA recall ≥ 0.70，semantic 加权 κ ≥ 0.60。
5. 不通过就改 prompt 或手册，最多两轮。仍不通过，LLM 只做 triage，500 对全部人工标。

### 阶段 B：Pilot 500 对

1. 四模型各跑 500 对，保存原始 JSON 和解析后的标签。
2. 多数票（≥3/4）作为预标；2–2 平票、所有 FM/FVA 候选、confidence=1 的样本，全部人工确认。
3. 你确认时只做“同意 / 改写 / 补 evidence”，不重新读整篇文档。
4. 人工确认后的标签是 pilot 的最终标签。
5. 算 LLM–human 一致性，作为 validation 阶段能不能放宽审计比例的依据。

### 阶段 C：Validation 1,000 对

1. 四模型预标，多数票。
2. 人工确认全部 FM/FVA 候选，再随机抽 20% 确认其余标签。
3. 如果审计样本上 FM 的 precision ≥ 0.85、FVA 的 precision ≥ 0.85，剩余 80% 可以采用模型预标，但论文里必须写“LLM-assisted, 20% audited”。
4. 如果达不到，validation 也全部人工确认。
5. 7 天后随机抽 100 对重标，算 intra-annotator agreement。

## 3. 论文里怎么写

可以写：

> Due to annotator availability, a single human annotator labeled all pilot pairs and audited all functional-mismatch / valid-alternative candidates in the validation set. Four LLMs (DeepSeek, Qwen, GLM, GPT) independently pre-annotated the same pairs; we report LLM–human agreement and use the models only for pre-annotation and triage.

不能写：

- “inter-annotator agreement”或“two independent annotators”；
- “human-verified”但不写审计比例；
- 把 LLM 多数票说成 ground truth；
- 只报 LLM–LLM 一致性，不报对人工的一致性。

必须在 Limitations 里写：

- single human annotator；
- LLM pre-annotation may share biases with the retrieval/method models；
- 20% audit rate and per-class precision/recall；
- 结论只覆盖 SRA-Bench 的六个任务，不外推到所有 skill retrieval。

## 4. 需要报告的数字

| 指标 | 用途 |
|---|---|
| 人工 7 天自复查 κ（100 对） | 替代双人 IAA，说明人工标签稳定 |
| 每个模型 vs 人工的 semantic / coverage 加权 κ | 模型是否能稳定复现人工判断 |
| 每个模型 vs 人工的 FM F1、FVA recall | 预标能不能用 |
| 四模型多数票 vs 人工的 FM F1、FVA recall | 最终预标方案 |
| FM/FVA 的审计 precision | validation 放宽审计比例的依据 |
| 每个任务的 LLM–human 一致性 | 判断哪些任务必须全人工 |

## 5. 成本与时间

- 30 对 × 4 模型：约 120 次调用，几分钟，成本不到 1 美元。
- 500 对 × 4 模型：约 2,000 次调用，单模型 0.7–1M token，四个模型合计约 3–5 美元。
- 1,000 对 × 4 模型：约 4,000 次调用，合计约 6–10 美元。
- 人工时间：校准 1h，pilot 6–8h，validation 6–9h，自复查 1.5h，合计 14–19h。

## 6. 关键限制

- LLM 只看到 1,500 字正文片段，和人工看到的一样。片段不够时，人工可以查全文，LLM 不能。这个差异要在一致性分析里说明。
- 四个模型如果对同一类错误有相同偏好，多数票不会纠正。所以必须保留人工确认，尤其是 FM/FVA。
- prompt 一旦冻结，不再按结果改。所有模型用同一版 prompt 和同一份输入。
- 模型输出里的 evidence 必须能定位到原文；定位不到就标为无效预标，交人工。
