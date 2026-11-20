# 结果文件

| 文件 | 内容 |
|---|---|
| `e0_dense_v100.json` | 六任务 Dense 的完整评测输出（config、per_category、overall、summary_tables） |
| `e0_skillsight_v100.json` | 六任务 SkillSight 的完整评测输出 |
| `e1_bm25.json` | BM25 分任务 R@1/5/10 与 MRR |
| `e1_rerank.json` | bge-reranker-v2-m3 在 Dense top-100 上的重排结果（跑完后提交） |
| `pilot/dense_top100.npz` | 5,400 条 query 的 top-100 索引、分数、任务、gold rank |
| `pilot/dense_top5.jsonl.gz` | 5,400 条 query 的 top-5 详细信息，含 skill 名、描述、正文前 1,500 字 |

## 当前主结果

R@10，micro：

| 方法 | R@10 | MRR |
|---|---:|---:|
| BM25 | 64.04 | 0.5179 |
| Dense | 66.04 | 0.5713 |
| SkillSight | 86.07 | 0.7390 |

分任务结果见根目录 `README.md`。

## 复现

`e0_dense_v100.json` 与 `e0_skillsight_v100.json` 由 `repo.cli` 生成；`e1_bm25.json` 由 `code/pilot/bm25_eval.py` 生成；`pilot/*` 由 `code/pilot/export_dense_topk.py` 生成。命令见 `docs/server_v100_setup.md`。

模型权重、SRA-Bench 原始语料、doc embedding 缓存不在仓库里，按需下载或重新生成。
