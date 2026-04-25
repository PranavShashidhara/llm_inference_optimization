"""
Block 1 — Baseline Inference Engine

Intentional bottlenecks (control system):
  - Full QKᵀ materialization: O(T²) memory at each decode step
  - Python-level decode loop: one token per iteration, no batching
  - No kernel fusion, no custom ops
  - Naive KV cache with torch.cat appends

This is V0. Every number it produces is a benchmark baseline.
"""

import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from typing import Optional

from model.kv_cache import NaiveKVCache


# ---------------------------------------------------------------------------
# Profiling containers
# ---------------------------------------------------------------------------

@dataclass
class DecodeStep:
    step: int
    token_id: int
    latency_ms: float
    kv_bytes: int


@dataclass
class InferenceTrace:
    prompt: str
    prefill_latency_ms: float = 0.0
    decode_steps: list = field(default_factory=list)
    total_tokens: int = 0

    def tokens_per_sec(self) -> float:
        if not self.decode_steps:
            return 0.0
        total_ms = sum(s.latency_ms for s in self.decode_steps)
        return len(self.decode_steps) / (total_ms / 1000.0)

    def p50_latency_ms(self) -> float:
        latencies = sorted(s.latency_ms for s in self.decode_steps)
        if not latencies:
            return 0.0
        return latencies[len(latencies) // 2]

    def p95_latency_ms(self) -> float:
        latencies = sorted(s.latency_ms for s in self.decode_steps)
        if not latencies:
            return 0.0
        idx = int(0.95 * len(latencies))
        return latencies[min(idx, len(latencies) - 1)]

    def summary(self) -> dict:
        return {
            "tokens_generated": len(self.decode_steps),
            "prefill_ms": round(self.prefill_latency_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec(), 2),
            "latency_p50_ms": round(self.p50_latency_ms(), 2),
            "latency_p95_ms": round(self.p95_latency_ms(), 2),
            "kv_peak_mb": round(
                max((s.kv_bytes for s in self.decode_steps), default=0) / (1024**2), 3
            ),
        }


# ---------------------------------------------------------------------------
# Naive attention (no flash, no tiling)
# ---------------------------------------------------------------------------

def naive_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Full QKᵀ materialization — O(T²) memory.

    Args:
        query: [batch, heads, q_len, head_dim]
        key:   [batch, heads, kv_len, head_dim]
        value: [batch, heads, kv_len, head_dim]

    Returns:
        output: [batch, heads, q_len, head_dim]

    Bottleneck: the full scores matrix is materialized and held in VRAM.
    At T=2048 this is 2048*2048*num_heads*4 bytes — significant on Jetson Orin.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    # [batch, heads, q_len, kv_len] — full materialization
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Causal mask (only needed during prefill; decode has q_len=1)
    if query.shape[2] > 1:
        q_len, kv_len = query.shape[2], key.shape[2]
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, value)


# ---------------------------------------------------------------------------
# Baseline inference engine
# ---------------------------------------------------------------------------

class BaselineInferenceEngine:
    """
    V0 baseline engine. No optimizations. Intentional overhead preserved.

    Usage:
        engine = BaselineInferenceEngine("Qwen/Qwen2-0.5B", device)
        trace = engine.generate("Hello world", max_new_tokens=100)
        print(trace.summary())
    """

    def __init__(self, model_name: str, device: torch.device, dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        print(f"[Block1] Loading tokenizer from {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[Block1] Loading model ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=str(device),
        )
        self.model.eval()

        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // self.num_heads

        print(f"[Block1] Ready — {self.num_layers} layers, "
              f"{self.num_heads} heads, head_dim={self.head_dim}")

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: NaiveKVCache,
    ) -> torch.Tensor:
        """
        Run a full forward pass on the prompt. Populates the KV cache.

        Returns logits for the last token position.
        """
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )

        # Populate the cache from HuggingFace past_key_values
        if out.past_key_values is not None:
            for layer_idx, (k, v) in enumerate(out.past_key_values):
                kv_cache.cache[layer_idx] = (k, v)
            kv_cache.stats.record(0, kv_cache.cache[0][0], kv_cache.cache[0][1])

        return out.logits[:, -1, :]  # [batch, vocab_size]

    # ------------------------------------------------------------------
    # Single decode step
    # ------------------------------------------------------------------

    def _decode_step(
        self,
        token_id: torch.Tensor,
        kv_cache: NaiveKVCache,
    ) -> torch.Tensor:
        """
        One token → logits for next token. Uses model's built-in cache handling.
        KV cache updated in-place via NaiveKVCache wrapper.

        Returns logits: [1, vocab_size]
        """
        # Build past_key_values from our naive cache
        past_kv = None
        if kv_cache.cache[0] is not None:
            past_kv = tuple(
                (kv_cache.cache[i][0], kv_cache.cache[i][1])
                for i in range(self.num_layers)
            )

        with torch.no_grad():
            out = self.model(
                input_ids=token_id.unsqueeze(0),
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )

        # Update cache with new keys/values (naive append)
        for layer_idx, (k, v) in enumerate(out.past_key_values):
            kv_cache.cache[layer_idx] = (k, v)

        # Track step for stats
        if kv_cache.cache[0] is not None:
            kv_cache.stats.record(
                kv_cache._step,
                kv_cache.cache[0][0],
                kv_cache.cache[0][1]
            )
            kv_cache._step += 1

        return out.logits[:, -1, :]  # [1, vocab_size]

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> InferenceTrace:
        """
        Full autoregressive generation with profiling.

        Args:
            prompt:         Input text
            max_new_tokens: Hard cap on generated tokens
            temperature:    Sampling temperature (ignored if greedy=True)
            greedy:         Use argmax sampling

        Returns:
            InferenceTrace with latency, throughput, and KV memory stats
        """
        trace = InferenceTrace(prompt=prompt)
        kv_cache = NaiveKVCache(self.num_layers, self.device)

        # --- Tokenize ---
        inputs = self.tokenizer(
            prompt, return_tensors="pt"
        ).to(self.device)
        input_ids = inputs["input_ids"]  # [1, prompt_len]

        # --- Prefill ---
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = self._prefill(input_ids, kv_cache)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        trace.prefill_latency_ms = (time.perf_counter() - t0) * 1000.0

        # --- Decode loop (Python-level, one token at a time) ---
        next_token = logits.argmax(dim=-1)  # [1]
        generated = [next_token.item()]

        eos_token_id = self.tokenizer.eos_token_id

        for step in range(max_new_tokens - 1):
            kv_bytes = sum(
                (e[0].element_size() * e[0].nelement() +
                 e[1].element_size() * e[1].nelement())
                for e in kv_cache.cache if e is not None
            )

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t_step = time.perf_counter()

            logits = self._decode_step(next_token, kv_cache)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            step_ms = (time.perf_counter() - t_step) * 1000.0

            if greedy:
                next_token = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            token_id = next_token.item()
            generated.append(token_id)

            trace.decode_steps.append(DecodeStep(
                step=step,
                token_id=token_id,
                latency_ms=step_ms,
                kv_bytes=kv_bytes,
            ))

            if token_id == eos_token_id:
                break

        trace.total_tokens = len(generated)
        return trace

    def decode_text(self, trace: InferenceTrace) -> str:
        token_ids = [s.token_id for s in trace.decode_steps]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)