# V100 环境与复现记录

这份文档记录在 Tesla V100（SM 7.0）上跑通 SkillSight / SRA-Bench 的环境、补丁和踩坑。目标是让别人拿到仓库后能复现 66.04 / 86.07 这两个数。

## 1. 硬件与系统

| 项目 | 配置 |
|---|---|
| GPU | 4× Tesla V100-SXM2-32GB，算力 7.0 |
| 驱动 | 580.167.08，显示 CUDA 13.0 |
| 系统 | Ubuntu 22.04.3 LTS，20 核，226G 内存 |
| Python | 3.11.18 |
| PyTorch | 2.6.0+cu124 |
| 关键包 | numpy 2.4.6，sentence-transformers 5.5.1，transformers 4.57.6 |

没有 `nvcc` 也可以跑。PyTorch wheel 自带 CUDA runtime，`torch.cuda.is_available()` 为 True 即可。

## 2. 环境创建

服务器上已有带 torch 2.6.0+cu124 的 conda 环境。为避免影响别人的环境，用 `--system-site-packages` 建独立 venv，只装增量包：

```bash
BASE=/path/to/conda/envs/<env-with-torch-2.6>/bin/python
python -m venv --system-site-packages /root/skillret/venv
/root/skillret/venv/bin/pip install sentence-transformers==5.5.1
/root/skillret/venv/bin/pip install xformers==0.0.29.post2
/root/skillret/venv/bin/pip install bm25s==0.3.16 openpyxl
```

如果从零建环境，torch 用官方 cu124 索引：

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## 3. V100 的两个硬限制

### 3.1 没有可用的 BF16

Qwen3-Embedding 的 `config.json` 默认 `torch_dtype: bfloat16`。V100 没有原生 BF16，直接加载会走模拟或报错。处理方式是在 `encoder.py` 里按算力选 dtype：

```python
def _cuda_dtype(device: str) -> torch.dtype:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16
```

`SentenceTransformer` 的 `model_kwargs` 也要显式传 `torch_dtype`，否则它会按 config 加载 BF16。

### 3.2 GQA + SDPA 在 SM 7.0 上会退回 math 后端

Qwen3 是 GQA：16 个 query 头、8 个 KV 头。V100 上 `scaled_dot_product_attention` 的 memory-efficient 内核不接受这种广播，会退回 math 后端，把完整 attention 矩阵物化。

- 8k、batch 4：申请 16 GiB，OOM。
- 32k、batch 2：申请 128 GiB，OOM。

修复分两步：

1. 关闭 SDPA 的 GQA 路径，让 transformers 先把 KV 展开到 16 头：

```python
from transformers.integrations import sdpa_attention
sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
```

2. 强制 memory-efficient 后端，禁用 math 和 flash：

```python
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

改完后 8k、batch 4 的显存峰值约 5.9G，可以正常跑。`code/patches/encoder_v100.patch` 是完整的 diff。

### 3.3 上下文长度

对 26,262 篇 skill 统计 token 长度：

| 分位数 | token |
|---|---:|
| 中位数 | 1,536 |
| p90 | 3,914 |
| p95 | 5,417 |
| p99 | 9,970 |
| 最大 | 245,054 |

只有 1.9% 超过 8k，0.11% 超过 16k。主实验用 `max_seq_length=8192`，Dense R@10 为 66.04，和 32k 下的 66.09 差 0.05pp。不要为了“和论文一致”强行上 32k，V100 上跑不动。

## 4. 模型下载

服务器直连 HuggingFace 会超时，但 `hf-mirror.com` 可用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='models/Qwen3-Embedding-0.6B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', local_dir='models/bge-reranker-v2-m3')"
```

Qwen3-Embedding 的本项目使用 revision 为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。

## 5. 复现命令

假设代码在 `/root/skillret/repo`，数据在 `third_party/SR-Agents-main`，缓存目录 `/root/skillret/cache`。

### 5.1 全量 Dense

```bash
cd /root/skillret
SNAP=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/ | head -1)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PYTHONPATH=/root/skillret python -m repo.cli \
  --benchmark sra \
  --repo-root /root/skillret/repo/third_party/SR-Agents-main \
  --model-path "$SNAP" \
  --device cuda:0 \
  --datasets theoremqa logicbench toolqa medcalcbench champ bigcodebench \
  --top-k 1 5 10 --candidate-k 300 --max-seq-length 8192 --doc-batch 4 \
  --variants dense \
  --output-dir runs/e0_dense_v100 --cache-dir /root/skillret/cache
```

### 5.2 多卡分片编码

26k 篇单卡约 3–4 小时，四张卡分片约 1 小时。`code/pilot/encode_shard.py` 按 `--start/--end` 编码一段，`merge_shards.py` 按顺序合并并写 meta。合并后的缓存文件名要和 CLI 期望的一致：

```text
all_<model_tag>_<max_seq_length>.npy
```

`model_tag` 是模型目录名，例如 `97b0c614...`。

### 5.3 SkillSight

```bash
PYTHONPATH=/root/skillret python -m repo.cli ... --variants skillsight \
  --output-dir runs/e0_skillsight_v100
```

### 5.4 BM25

```bash
PYTHONPATH=/root/skillret/repo python pilot/bm25_eval.py \
  --repo-root .../SR-Agents-main --out runs/e1_bm25.json
```

### 5.5 Reranker

```bash
PYTHONPATH=/root/skillret/repo python pilot/rerank_eval.py \
  --repo-root .../SR-Agents-main \
  --top-npz runs/pilot/dense_top100.npz \
  --top5-jsonl runs/pilot/dense_top5.jsonl \
  --model-path models/bge-reranker-v2-m3 \
  --out runs/e1_rerank.json --batch-size 64 --max-length 512
```

## 6. 期望结果

| 方法 | R@10 | 备注 |
|---|---:|---|
| Dense | 66.04 | 对照 4090/32k 的 66.09 |
| SkillSight | 86.07 | 对照 86.13 |
| BM25 | 64.04 | 分任务与 Dense 反转明显 |

机制侧：背景子空间秩 2，解释方差 93.36%，Otsu IDF 阈值 5.5999，通用 token 2,791 个。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `CUDA out of memory`，申请 16/128 GiB | SDPA 走 math 后端 | 应用 3.2 的补丁，确认 8k、batch 4 |
| `No available kernel` | 强制 flash 后端但 SM 7.0 不支持 | 只启用 memory-efficient，不要启用 flash |
| `bfloat16` 报错或结果异常 | V100 没有原生 BF16 | 显式用 FP16 |
| `git clone`/`pip` 访问 HuggingFace 超时 | 网络限制 | 用 `HF_ENDPOINT=https://hf-mirror.com` |
| 多卡分片后合并，CLI 仍重新编码 | 缓存文件名或 meta 不匹配 | 检查 `model_tag`、`max_seq_length`、`text_sha1` |
| GPU 0 被别人占满 | 共享服务器 | `CUDA_VISIBLE_DEVICES=<free>`，不要抢卡 |

## 8. 未做的部分

- Qwen3-4B 的 Agent 端到端没有在 V100 上跑。根分区只剩约 6.6G，4B 权重约 8G，放不下；服务器也没有 vLLM。E7 默认删除。
- 没有跑 Llama-3.1-8B / GPT API 的完整矩阵。
- SkillRouter-Eval-Core 与 SkillRet 数据不在本机，外部验证留到后面。
