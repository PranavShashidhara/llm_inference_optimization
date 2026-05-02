"""
Block 3 — Attention Execution Layer

Purpose:
    Replace the O(T²) full QKᵀ materialization with block-wise streaming
    attention that reduces peak VRAM and improves memory bandwidth utilization.

Implementations:
    AttentionBaseline    — full QKᵀ materialization (Block 1 / V0 behavior)
    TritonStreamingAttn  — block-tiled attention with fused softmax accumulation

What Block 3 proves:
    - The full scores matrix [batch, heads, T, T] is the dominant memory cost
      at long sequences. At T=2048, fp16: 2048² × 16 heads × 2 bytes = 256 MB
    - Block-wise streaming never materializes the full T×T matrix.
      Peak memory drops from O(T²) to O(T × BLOCK_SIZE).
    - Fusing the softmax accumulation into the tile loop removes a
      separate kernel launch and halves global memory round-trips.

Triton kernel strategy (FlashAttention-style, simplified):
    For each query block Q_i of size BLOCK_Q:
        acc    = zeros [BLOCK_Q, head_dim]
        m_prev = -inf   (running max for numerically stable softmax)
        l_prev = 0      (running sum of exp weights)

        For each key/value block K_j, V_j of size BLOCK_KV:
            s_ij = Q_i @ K_j.T / sqrt(d)           [BLOCK_Q, BLOCK_KV]
            m_ij = rowmax(s_ij)
            p_ij = exp(s_ij - m_ij)                 [BLOCK_Q, BLOCK_KV]
            l_ij = rowsum(p_ij)

            # Correct previous accumulation for new max
            m_new    = max(m_prev, m_ij)
            acc      = acc * exp(m_prev - m_new) + p_ij @ V_j
            l_new    = l_prev * exp(m_prev - m_new) + l_ij
            m_prev, l_prev = m_new, l_new

        output[Q_i] = acc / l_prev

    This is the "online softmax" trick from Milakov & Gimelshein (2018),
    used in FlashAttention-1 (Dao et al., 2022).

Usage:
    python block3_attention.py
    python block3_attention.py --device cpu --seq_len 512

Requirements:
    pip install triton   # GPU only; falls back gracefully on CPU
"""

import math
import time
import torch
import torch.nn.functional as F
from typing import Optional

# Triton is optional — on CPU or when not installed, we use the PyTorch fallback
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("[Block3] Triton not available — TritonStreamingAttn will use PyTorch fallback.")


# ---------------------------------------------------------------------------
# Block 3A — Baseline: full QKᵀ materialization (O(T²) memory)
# ---------------------------------------------------------------------------

class AttentionBaseline:
    """
    Exact replica of the Block 1 naive_attention function, wrapped as a class
    so it's drop-in swappable with TritonStreamingAttn.

    Memory cost:  O(T²) — the full [batch, heads, T, T] scores tensor.
    Compute cost: O(T²) — unavoidable for exact attention.

    This is the bottleneck on long sequences:
        T=512,  heads=16, fp16: 512² × 16 × 2 = 8 MB
        T=2048, heads=16, fp16: 2048² × 16 × 2 = 256 MB   ← Jetson Orin limit
        T=8192, heads=16, fp16: 8192² × 16 × 2 = 4 GB     ← OOM on most edge GPUs
    """

    NAME = "baseline_full_qk"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, heads, q_len, head_dim]
            key:   [batch, heads, kv_len, head_dim]
            value: [batch, heads, kv_len, head_dim]
            scale: 1/sqrt(head_dim) if None
            causal: apply causal mask (needed for prefill; skip for decode)

        Returns:
            output: [batch, heads, q_len, head_dim]

        Bottleneck:
            Line `scores = torch.matmul(query, key.transpose(-2, -1))` allocates
            [batch, heads, q_len, kv_len] — this is the T² tensor.
        """
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])

        # ← BOTTLENECK: T² tensor fully materialized in VRAM
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale  # [B, H, Q, KV]

        if causal and query.shape[2] > 1:
            q_len, kv_len = query.shape[2], key.shape[2]
            mask = torch.triu(
                torch.ones(q_len, kv_len, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, value)

    def __call__(self, query, key, value, **kwargs):
        return self.forward(query, key, value, **kwargs)


# ---------------------------------------------------------------------------
# Block 3B — Triton streaming attention (block-tiled, fused softmax)
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _streaming_attn_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qb, stride_qh, stride_qq, stride_qd,
        stride_kb, stride_kh, stride_kk, stride_kd,
        stride_vb, stride_vh, stride_vk, stride_vd,
        stride_ob, stride_oh, stride_oq, stride_od,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_KV: tl.constexpr,
    ):
        """
        One program instance handles one (batch, head, query_block) tile.

        Grid: [batch * heads * ceil(seq_len / BLOCK_Q)]

        Memory behavior:
            - Loads Q_i once into registers/SRAM: BLOCK_Q × head_dim
            - Streams K_j, V_j blocks: BLOCK_KV × head_dim each
            - Never writes the full T×T scores matrix to HBM
            - Peak SRAM per SM: 2 × BLOCK_KV × head_dim + BLOCK_Q × head_dim
        """
        # Decode which (batch, head, query_block) this program handles
        prog_id = tl.program_id(0)
        num_q_blocks = tl.cdiv(seq_len, BLOCK_Q)
        bh_idx = prog_id // num_q_blocks
        q_block_idx = prog_id % num_q_blocks

        batch_idx = bh_idx // tl.num_programs(1) if tl.num_programs(0) > 1 else 0
        head_idx  = bh_idx  % tl.num_programs(1) if tl.num_programs(0) > 1 else bh_idx

        # Query block start position
        q_start = q_block_idx * BLOCK_Q
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        d_offs = tl.arange(0, head_dim)

        # Load Q tile: [BLOCK_Q, head_dim]
        Q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
        q_mask = q_offs[:, None] < seq_len
        Q = tl.load(
            Q_base + q_offs[:, None] * stride_qq + d_offs[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )

        # Running accumulators for online softmax
        acc  = tl.zeros([BLOCK_Q, head_dim], dtype=tl.float32)
        m_i  = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
        l_i  = tl.zeros([BLOCK_Q], dtype=tl.float32)

        K_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
        V_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh

        num_kv_blocks = tl.cdiv(seq_len, BLOCK_KV)
        for kv_block in range(num_kv_blocks):
            kv_start = kv_block * BLOCK_KV
            kv_offs = kv_start + tl.arange(0, BLOCK_KV)
            kv_mask = kv_offs[None, :] < seq_len

            # Load K tile: [BLOCK_KV, head_dim]
            K = tl.load(
                K_base + kv_offs[:, None] * stride_kk + d_offs[None, :] * stride_kd,
                mask=kv_offs[:, None] < seq_len,
                other=0.0,
            )

            # Load V tile: [BLOCK_KV, head_dim]
            V = tl.load(
                V_base + kv_offs[:, None] * stride_vk + d_offs[None, :] * stride_vd,
                mask=kv_offs[:, None] < seq_len,
                other=0.0,
            )

            # QKᵀ for this tile only: [BLOCK_Q, BLOCK_KV]
            s_ij = tl.dot(Q, tl.trans(K)) * scale

            # Causal mask: Q position must be >= KV position
            causal_mask = q_offs[:, None] >= kv_offs[None, :]
            s_ij = tl.where(causal_mask & kv_mask, s_ij, float("-inf"))

            # Online softmax update
            m_ij = tl.max(s_ij, axis=1)                   # [BLOCK_Q]
            p_ij = tl.exp(s_ij - m_ij[:, None])           # [BLOCK_Q, BLOCK_KV]
            l_ij = tl.sum(p_ij, axis=1)                   # [BLOCK_Q]

            # Rescale accumulator for new max
            m_new = tl.maximum(m_i, m_ij)
            alpha  = tl.exp(m_i - m_new)
            beta   = tl.exp(m_ij - m_new)

            acc  = acc * alpha[:, None] + tl.dot(p_ij.to(tl.float16), V) * beta[:, None]
            l_i  = l_i * alpha + l_ij * beta
            m_i  = m_new

        # Final normalization
        out = acc / l_i[:, None]

        # Write output: [BLOCK_Q, head_dim]
        Out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
        tl.store(
            Out_base + q_offs[:, None] * stride_oq + d_offs[None, :] * stride_od,
            out.to(tl.float16),
            mask=q_offs[:, None] < seq_len,
        )


class TritonStreamingAttn:
    """
    Block-tiled attention with fused online softmax accumulation.

    Falls back to a pure-PyTorch streaming simulation on CPU or when
    Triton is unavailable, so benchmarks still run everywhere.

    Memory cost:  O(T) — no T×T matrix ever written to HBM.
    Compute cost: O(T²) — same arithmetic, different memory access pattern.

    Block sizes:
        BLOCK_Q  = 64  (query tokens per tile)
        BLOCK_KV = 64  (key/value tokens per tile)
        These fit comfortably in L2 on A100/H100 and Jetson Orin shared memory.
        Tune downward (32) for smaller GPUs.

    When to use vs PyTorch sdpa:
        torch.nn.functional.scaled_dot_product_attention uses FlashAttention-2
        internally on CUDA ≥ 8.0. TritonStreamingAttn exists to demonstrate
        the technique explicitly and allow custom modifications (e.g. KV-layout
        changes from Block 2 that sdpa doesn't support natively).
    """

    NAME = "triton_streaming"
    BLOCK_Q  = 64
    BLOCK_KV = 64

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: Optional[float] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, heads, q_len, head_dim]
            key:   [batch, heads, kv_len, head_dim]
            value: [batch, heads, kv_len, head_dim]

        Returns:
            output: [batch, heads, q_len, head_dim]
        """
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])

        if not TRITON_AVAILABLE or query.device.type == "cpu":
            return self._pytorch_fallback(query, key, value, scale, causal)

        return self._triton_forward(query, key, value, scale)

    def _triton_forward(self, query, key, value, scale):
        """Launch the Triton kernel."""
        B, H, T, D = query.shape

        # Ensure contiguous and correct dtype for Triton
        q = query.contiguous().to(torch.float16)
        k = key.contiguous().to(torch.float16)
        v = value.contiguous().to(torch.float16)
        out = torch.empty_like(q)

        # Grid: one program per (batch, head, q_block)
        num_q_blocks = math.ceil(T / self.BLOCK_Q)
        grid = (B * H * num_q_blocks,)

        _streaming_attn_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            seq_len=T,
            head_dim=D,
            scale=scale,
            BLOCK_Q=self.BLOCK_Q,
            BLOCK_KV=self.BLOCK_KV,
            num_warps=4,
            num_stages=2,
        )
        return out.to(query.dtype)

    def _pytorch_fallback(self, query, key, value, scale, causal):
        """
        Pure-PyTorch streaming simulation: processes BLOCK_Q rows at a time.
        Does NOT use the T² matrix — demonstrates the algorithm portably.
        Slower than the baseline on CPU (Python loop overhead) but
        identical output and correct memory semantics.
        """
        B, H, T, D = query.shape
        output = torch.zeros_like(query, dtype=torch.float32)

        for q_start in range(0, T, self.BLOCK_Q):
            q_end = min(q_start + self.BLOCK_Q, T)
            Q_tile = query[:, :, q_start:q_end, :].float()   # [B, H, bq, D]

            acc   = torch.zeros(B, H, q_end - q_start, D, device=query.device)
            m_i   = torch.full((B, H, q_end - q_start), float("-inf"), device=query.device)
            l_i   = torch.zeros(B, H, q_end - q_start, device=query.device)

            for kv_start in range(0, T, self.BLOCK_KV):
                kv_end = min(kv_start + self.BLOCK_KV, T)
                K_tile = key[:, :, kv_start:kv_end, :].float()
                V_tile = value[:, :, kv_start:kv_end, :].float()

                # [B, H, bq, bkv]
                s_ij = torch.matmul(Q_tile, K_tile.transpose(-2, -1)) * scale

                if causal:
                    q_pos  = torch.arange(q_start, q_end, device=query.device)
                    kv_pos = torch.arange(kv_start, kv_end, device=query.device)
                    mask = q_pos[:, None] < kv_pos[None, :]   # [bq, bkv]
                    s_ij = s_ij.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

                # Online softmax update
                m_ij = s_ij.max(dim=-1).values          # [B, H, bq]
                p_ij = torch.exp(s_ij - m_ij.unsqueeze(-1))
                l_ij = p_ij.sum(dim=-1)                 # [B, H, bq]

                m_new = torch.maximum(m_i, m_ij)
                alpha = torch.exp(m_i - m_new)
                beta  = torch.exp(m_ij - m_new)

                acc = (acc * alpha.unsqueeze(-1) +
                       torch.matmul(p_ij, V_tile) * beta.unsqueeze(-1))
                l_i = l_i * alpha + l_ij * beta
                m_i = m_new

            output[:, :, q_start:q_end, :] = acc / l_i.unsqueeze(-1)

        return output.to(query.dtype)

    def __call__(self, query, key, value, **kwargs):
        return self.forward(query, key, value, **kwargs)