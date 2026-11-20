# 数据说明

## annotations

| 文件 | 内容 |
|---|---|
| `annotation_pilot.csv` / `.xlsx` | 100 条 query × top-5，共 500 对，盲标表 |
| `annotation_pilot_key.csv` | pair_id 对应的 skill_id、gold、Dense 排名/分数，标注时不要打开 |
| `annotation_validation.csv` / `.xlsx` | 200 条 query × top-5，共 1,000 对 |
| `annotation_validation_key.csv` | 同上，分析用 |

Excel 版带下拉选项：`semantic` 0/1/2，`coverage` 0/1/2，`judgment` Gold/FVA/Invalid，`invalid_reason` generic/topic_related/unrelated/duplicate，`confidence` 1/2/3。

完整规则见 `docs/02_标注手册.md`。

## samples

| 文件 | 内容 |
|---|---|
| `pilot_sample.csv` | 300 条抽样清单（pilot 100 + validation 200），含 split、task、stratum、hit@5、gold_rank_dense |
| `sample_key.json` | 每个 split/task 下失败与成功 query 的 ID 列表，用于复核抽样 |

抽样种子 `20261120`。pilot 实际 71 失败 / 29 成功；validation 实际 144 失败 / 56 成功。medcalcbench 的 Dense R@5 为 98.36%，失败样本不足，validation 的 33 个失败名额用成功样本补齐。

分析时不要直接用样本比例估计总体，按任务真实失败率回加权：

```text
P(FM) = Σ_t w_t · P(fail | t) · P(FM | fail, t)
w_t = N_t / 5400
```
