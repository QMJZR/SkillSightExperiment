# pilot 脚本

| 脚本 | 作用 |
|---|---|
| `encode_shard.py` | 按 `--start/--end` 编码语料的一段，供 4 卡并行 |
| `merge_shards.py` | 按语料顺序合并分片，写 CLI 能识别的 `.npy` 与 `.meta.json` |
| `bm25_eval.py` | 26,262 篇语料的 BM25 检索与分任务指标 |
| `export_dense_topk.py` | 导出 5,400 条 query 的 Dense top-100 与 top-5（含 skill 文本片段） |
| `sample_queries.py` | 按任务 × 成功/失败分层抽样，固定种子 20261120 |
| `make_annotation_sheet.py` | 生成盲标 CSV 与分析用 key |
| `make_annotation_xlsx.py` | 把 CSV 转成带下拉选项和说明页的 Excel |
| `rerank_eval.py` | bge-reranker-v2-m3 在 Dense top-100 上重排 |

所有脚本都从仓库根目录运行，并设置：

```bash
export PYTHONPATH=/path/to/repo
```

`encode_shard.py` 与 `merge_shards.py` 的缓存文件名必须和 CLI 的 `cache.py` 约定一致，参见 `docs/server_v100_setup.md` 的 5.2。
