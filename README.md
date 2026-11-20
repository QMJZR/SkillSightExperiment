# SkillSightExperiment

LLM Agent Skill Retrieval 的失败模式与 Functional Mismatch 研究。当前阶段是 Milestone 1：在 SRA-Bench 上建立 BM25 / Dense / Dense+Reranker 基线，抽样标注失败类型，估计 Functional Mismatch 的占比与排序错误率。

研究计划见 [`docs/01_最终实验计划.md`](docs/01_最终实验计划.md)，标注规则见 [`docs/02_标注手册.md`](docs/02_标注手册.md)。

## 当前状态

更新时间：2026-11-20

| 阶段 | 状态 | 结果 |
|---|---|---|
| 环境与复现（G0） | 通过 | Dense R@10 66.04（对照 66.09）；SkillSight 86.07（对照 86.13） |
| E1 基线 | 3/4 | BM25 64.04；Reranker 运行中 |
| Pilot 抽样 | 完成 | 100 query / 500 pair |
| Validation 抽样 | 完成 | 200 query / 1,000 pair |
| 标注手册 | v1.0 | `docs/02_标注手册.md` |
| Pilot 标注 | 未开始 | 先做 30 对校准 |

## 主要结果

R@10，micro，SRA-Bench 5,400 条 query、26,262 个候选 skill：

| 任务 | BM25 | Dense | SkillSight |
|---|---:|---:|---:|
| theoremqa | 85.81 | 94.78 | 96.12 |
| logicbench | 26.84 | 35.39 | 69.21 |
| toolqa | 80.56 | 50.21 | 91.54 |
| medcalcbench | 63.18 | 99.36 | 99.00 |
| champ | 45.98 | 62.94 | 58.98 |
| bigcodebench | 58.21 | 55.94 | 76.68 |
| **总体** | **64.04** | **66.04** | **86.07** |

两个现象需要保留：

1. BM25 和 Dense 的分任务结果出现双向反转，BM25 在 toolqa 上高 30.35pp，Dense 在 medcalcbench 上高 36.18pp。失败分析不能只用一个 retriever 的失败集代表“现有方法”。
2. SkillSight 在 champ 和 medcalcbench 上略低于 Dense，最强基线不是逐任务最强，所有对比都要分任务报。

## 环境

- 4× Tesla V100-SXM2-32GB（SM 7.0）
- Ubuntu 22.04，Python 3.11，torch 2.6.0+cu124
- FP16 推理；V100 不支持实用的 BF16 和 FlashAttention
- 主实验 `max_seq_length=8192`。语料 p99 约 9,970 token，1.9% 文档被截断；32k 在 V100 上会触发 math SDPA 并 OOM

完整环境、补丁和踩坑记录见 [`docs/server_v100_setup.md`](docs/server_v100_setup.md)。

## 仓库结构

```text
.
├── code/
│   ├── encoder_v100.py          # 为 V100 改过的 DenseEncoder
│   ├── patches/                 # 对上游 SkillSight 的 patch
│   └── pilot/                   # 抽样、导出、BM25、Reranker、标注表脚本
├── data/
│   ├── annotations/             # pilot / validation 标注表与 key
│   └── samples/                 # 分层抽样清单
├── docs/
│   ├── 01_最终实验计划.md
│   ├── 02_标注手册.md
│   └── server_v100_setup.md
├── logs/
├── results/
│   ├── e0_dense_v100.json
│   ├── e0_skillsight_v100.json
│   ├── e1_bm25.json
│   └── pilot/                   # top-100 索引与压缩后的 top-5
└── README.md
```

## 复现步骤

1. 下载数据：`WeihangSu/SRA-Bench`（5,400 query，26,262 skill）。
2. 下载模型：`Qwen/Qwen3-Embedding-0.6B`，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。
3. 以上游 `QMJZR/SkillSight` 为基础，应用 `code/patches/encoder_v100.patch`。
4. 安装 `code/requirements-v100.txt`，按 `docs/server_v100_setup.md` 设置 FP16 和 SDPA。
5. 把 `code/pilot/` 放到仓库的 `pilot/` 下，按下面的顺序跑：

```bash
# 全量 Dense / SkillSight
python -m repo.cli --benchmark sra --variants dense skillsight ...

# BM25
python pilot/bm25_eval.py --repo-root ... --out results/e1_bm25.json

# Dense top-100 与 top-5 导出
python pilot/export_dense_topk.py ...

# 分层抽样与标注表
python pilot/sample_queries.py ...
python pilot/make_annotation_sheet.py ...
python pilot/make_annotation_xlsx.py ...

# Reranker
python pilot/rerank_eval.py ...
```

具体参数见各脚本的 `--help` 和 `docs/01_最终实验计划.md` 的 E0–E5。

## 数据与文件说明

- `results/pilot/dense_top5.jsonl.gz`：5,400 条 query 的 Dense top-5，含 skill 名、描述、正文前 1,500 字，用于复现标注表。
- `results/pilot/dense_top100.npz`：Dense top-100 的索引与分数，用于复现 Reranker。
- `data/annotations/*_key.csv`：pair_id 到 skill_id、gold、Dense 排名与分数的映射。标注时应保持盲标，不要打开。
- 以下内容不放进仓库，按需重新生成：Qwen3-Embedding 权重、bge-reranker 权重、SRA-Bench 原始语料、doc embedding 缓存。

## 已知限制

- V100 上 32k 上下文不可行，主实验用 8k，1.9% 文档被截断；Dense 复现差 0.05pp，可以接受。
- medcalcbench 的 Dense R@5 为 98.36%，失败样本不足。validation 计划 160 失败 / 40 成功，实际为 144 / 56，分析时按任务真实失败率回加权。
- SkillSight 在 champ 上比 Dense 低 3.96pp，在 medcalcbench 上低 0.36pp。不能把 SkillSight 当成逐任务上界。
- 标注尚未开始，Functional Mismatch 的占比目前没有数据，不要提前写结论。

## 归属

- `code/pilot/` 为本项目新写。
- `code/encoder_v100.py` 修改自 `QMJZR/SkillSight` 的 `encoder.py`，上游许可见原仓库。
- SRA-Bench、Qwen3-Embedding、bge-reranker 归各自作者所有。
