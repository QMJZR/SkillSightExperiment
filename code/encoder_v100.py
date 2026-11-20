"""Dense encoder wrapper used by SkillSight."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


SKILLROUTER_QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)


def _cuda_dtype(device: str) -> torch.dtype:
    """Use BF16 on Ampere and newer; V100 (SM 7.0) only has usable FP16."""
    if not device.startswith("cuda"):
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _configure_sdpa(device: str) -> None:
    """Prefer the memory-efficient SDPA kernel on pre-Ampere GPUs.

    PyTorch falls back to the math kernel for head_dim=128 on SM 7.0 unless the
    memory-efficient backend is selected explicitly. The math kernel materialises
    the full attention matrix and OOMs at 8k tokens.
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        return
    try:
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception as exc:
        print(f"[encoder] cannot force memory-efficient SDPA: {exc}", flush=True)
    # Qwen3 uses GQA. On SM 7.0 the fused kernel rejects 16 query / 8 KV heads,
    # so make transformers expand the KV heads before calling SDPA.
    try:
        from transformers.integrations import sdpa_attention

        sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key: False
    except Exception as exc:
        print(f"[encoder] cannot disable SDPA GQA path: {exc}", flush=True)


class DenseEncoder:
    def __init__(self, model_path: str, device: str = "cuda:0", max_seq_length: int = 4096):
        self.device = device
        self.max_seq_length = max_seq_length
        _configure_sdpa(device)
        local_path = Path(model_path)
        self.native_transformers = (
            local_path.is_dir()
            and (local_path / "config.json").exists()
            and not (local_path / "config_sentence_transformers.json").exists()
        )

        if self.native_transformers:
            from transformers import AutoModel, AutoTokenizer

            dtype = _cuda_dtype(device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
            )
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                attn_implementation="sdpa",
            ).to(device).eval()
            self.dim = int(self.model.config.hidden_size)
        else:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=True,
                model_kwargs={
                    "attn_implementation": "sdpa",
                    "torch_dtype": _cuda_dtype(device),
                },
            )
            self.model.max_seq_length = max_seq_length
            self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[encoder] loaded: {model_path}", flush=True)
        print(
            f"[encoder] dim={self.dim} max_seq_length={max_seq_length} "
            f"device={device} native_transformers={self.native_transformers} "
            f"dtype={_cuda_dtype(device)}",
            flush=True,
        )

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = True,
        prompt_name: Optional[str] = None,
    ) -> np.ndarray:
        if self.native_transformers:
            texts = list(texts)
            if prompt_name == "query":
                texts = [SKILLROUTER_QUERY_INSTRUCTION + text for text in texts]
            # Match SentenceTransformer's length-sorted batching behavior so
            # long and short documents do not share large padded batches.
            order = sorted(range(len(texts)), key=lambda idx: len(texts[idx]))
            ordered_texts = [texts[idx] for idx in order]
            batches = []
            for start in tqdm(
                range(0, len(ordered_texts), batch_size),
                desc="  encode",
                disable=not show_progress,
                unit="batch",
            ):
                encoded = self.tokenizer(
                    ordered_texts[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    hidden = self.model(**encoded).last_hidden_state
                    padding_is_left = bool(encoded["attention_mask"][:, -1].all())
                    if padding_is_left:
                        embeddings = hidden[:, -1]
                    else:
                        lengths = encoded["attention_mask"].sum(dim=1) - 1
                        row_ids = torch.arange(hidden.shape[0], device=hidden.device)
                        embeddings = hidden[row_ids, lengths]
                    embeddings = F.normalize(embeddings.float(), p=2, dim=1)
                batches.append(embeddings.cpu().numpy())
            encoded_embeddings = np.vstack(batches).astype(np.float32)
            restored = np.empty_like(encoded_embeddings)
            restored[order] = encoded_embeddings
            return restored

        emb = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name=prompt_name,
        )
        return emb.astype(np.float32)
