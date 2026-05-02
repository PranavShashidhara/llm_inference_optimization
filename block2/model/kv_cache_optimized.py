"""
Block 2 — Optimized KV Cache
Layout: coalesced memory + warp-friendly access patterns

Optimizations vs Block 1:
  ✓ Pre-allocated buffer (no torch.cat every step)
  ✓ Coalesced layout: [batch, num_heads, seq_len, head_dim]
      → enables contiguous global memory reads per warp
  ✓ Efficient indexing: O(1) updates via buffer pointer increment
  ✓ Memory-aware growth: allocate 2x current, reuse on underflow
  ✓ No fragmentation from repeated allocs/deallocs

Intended for decode phase where seq_len grows linearly.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizedKVCacheStats:
    """Tracks memory efficiency improvements."""
    steps: list = field(default_factory=list)
    bytes_allocated: list = field(default_factory=list)
    bytes_used: list = field(default_factory=list)
    num_reallocations: int = 0

    def record(self, step: int, allocated: int, used: int):
        self.steps.append(step)
        self.bytes_allocated.append(allocated)
        self.bytes_used.append(used)

    def record_reallocation(self):
        self.num_reallocations += 1

    def utilization_percent(self) -> float:
        """What fraction of allocated memory is actually used?"""
        if not self.bytes_allocated:
            return 0.0
        total_allocated = sum(self.bytes_allocated)
        total_used = sum(self.bytes_used)
        return (total_used / total_allocated * 100.0) if total_allocated > 0 else 0.0

    def growth_rate_bytes_per_step(self) -> float:
        """Linear growth rate of used memory."""
        if len(self.bytes_used) < 2:
            return 0.0
        deltas = [b - a for a, b in zip(self.bytes_used, self.bytes_used[1:])]
        return sum(deltas) / len(deltas) if deltas else 0.0

    def peak_mb(self) -> float:
        if not self.bytes_allocated:
            return 0.0
        return max(self.bytes_allocated) / (1024 ** 2)

    def wasted_mb(self) -> float:
        """Unused pre-allocated memory."""
        if not self.bytes_allocated or not self.bytes_used:
            return 0.0
        peak_allocated = max(self.bytes_allocated)
        peak_used = max(self.bytes_used)
        return (peak_allocated - peak_used) / (1024 ** 2)


class OptimizedKVCache:
    """
    KV cache with pre-allocated buffers and coalesced memory layout.

    Key insight:
      During decode, seq_len grows predictably: 1, 2, 3, ..., T.
      We can pre-allocate 2x the final size and reuse, avoiding expensive
      reallocs and fragmenting copies (torch.cat).

    Layout per layer:
      k_buffer: [batch, num_heads, max_seq_len, head_dim]  (pre-alloc)
      v_buffer: [batch, num_heads, max_seq_len, head_dim]  (pre-alloc)
      seq_len:  current position in buffer (0 to max_seq_len)

    Access pattern (decode):
      Step t:
        - write new k,v to k_buffer[:, :, seq_len, :] and v_buffer[:, :, seq_len, :]
        - increment seq_len
        - attention reads k_buffer[:, :, :seq_len, :] and v_buffer[:, :, :seq_len, :]

    Why this is fast:
      - Contiguous global memory reads: full rows in k/v can coalesce
      - No per-step allocations: reuse fixed buffers
      - No torch.cat: just indexing and increment
      - Warp-friendly: head_dim is naturally packed
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        initial_seq_len: int = 1,
        growth_factor: float = 2.0,
    ):
        """
        Args:
            num_layers:      number of transformer layers
            batch_size:      batch dimension
            num_heads:       number of attention heads
            head_dim:        dimension per head
            device:          torch device
            dtype:           torch dtype (usually float16)
            initial_seq_len: starting max_seq_len (usually 1 for decode-only)
            growth_factor:   when realloc needed, allocate growth_factor * current_max
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.growth_factor = growth_factor

        # Pre-alloc buffers for each layer
        # Start with 2x initial, so we get at least a few decode steps before realloc
        self.max_seq_len = max(initial_seq_len * 2, 128)  # min 128
        self.current_seq_len = 0

        self.k_buffers = []
        self.v_buffers = []

        for _ in range(num_layers):
            k_buf = torch.zeros(
                (batch_size, num_heads, self.max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            v_buf = torch.zeros(
                (batch_size, num_heads, self.max_seq_len, head_dim),
                dtype=dtype,
                device=device,
            )
            self.k_buffers.append(k_buf)
            self.v_buffers.append(v_buf)

        self.stats = OptimizedKVCacheStats()
        self._step = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new key/value to the cache for this layer.

        Args:
            layer_idx: which transformer layer
            key:   [batch, heads, 1, head_dim]  (single new token)
            value: [batch, heads, 1, head_dim]

        Returns:
            Full cached key and value tensors for attention computation.
            These are views into the pre-allocated buffers.
        """
        # Check if we need to grow buffers
        if self.current_seq_len >= self.max_seq_len:
            self._reallocate()

        # Write new k,v to buffer at current position
        self.k_buffers[layer_idx][:, :, self.current_seq_len, :] = key.squeeze(2)
        self.v_buffers[layer_idx][:, :, self.current_seq_len, :] = value.squeeze(2)

        # Record stats for layer 0 only
        if layer_idx == 0:
            self.current_seq_len += 1
            allocated_bytes = (
                self.k_buffers[0].element_size() * self.k_buffers[0].nelement()
                + self.v_buffers[0].element_size() * self.v_buffers[0].nelement()
            )
            used_bytes = (
                self.k_buffers[0].element_size() * self.batch_size * self.num_heads
                * self.current_seq_len * self.head_dim
                + self.v_buffers[0].element_size() * self.batch_size * self.num_heads
                * self.current_seq_len * self.head_dim
            )
            self.stats.record(self._step, allocated_bytes, used_bytes)
            self._step += 1

        # Return views of the full cached sequences
        return (
            self.k_buffers[layer_idx][:, :, :self.current_seq_len, :],
            self.v_buffers[layer_idx][:, :, :self.current_seq_len, :],
        )

    def _reallocate(self):
        """Grow buffers when seq_len exceeds max_seq_len."""
        old_max = self.max_seq_len
        self.max_seq_len = int(self.max_seq_len * self.growth_factor)
        self.stats.record_reallocation()

        new_k_buffers = []
        new_v_buffers = []

        for layer_idx in range(self.num_layers):
            # Allocate new buffers
            new_k = torch.zeros(
                (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            new_v = torch.zeros(
                (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )

            # Copy existing data
            new_k[:, :, :old_max, :] = self.k_buffers[layer_idx][:, :, :old_max, :]
            new_v[:, :, :old_max, :] = self.v_buffers[layer_idx][:, :, :old_max, :]

            new_k_buffers.append(new_k)
            new_v_buffers.append(new_v)

        self.k_buffers = new_k_buffers
        self.v_buffers = new_v_buffers

    def get(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get full cached k,v for a layer."""
        if self.current_seq_len == 0:
            return None
        return (
            self.k_buffers[layer_idx][:, :, :self.current_seq_len, :],
            self.v_buffers[layer_idx][:, :, :self.current_seq_len, :],
        )

    def seq_len(self) -> int:
        return self.current_seq_len

    def clear(self):
        """Reset cache for new sequence."""
        self.current_seq_len = 0
        self.max_seq_len = max(128, int(128 * self.growth_factor))  # reset to initial

        # Re-allocate buffers
        for i in range(self.num_layers):
            self.k_buffers[i] = torch.zeros(
                (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            self.v_buffers[i] = torch.zeros(
                (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )

        self.stats = OptimizedKVCacheStats()
        self._step = 0

    def memory_report(self) -> dict:
        """Detailed memory usage report."""
        allocated_bytes = 0
        used_bytes = 0

        for i in range(self.num_layers):
            allocated_bytes += (
                self.k_buffers[i].element_size() * self.k_buffers[i].nelement()
                + self.v_buffers[i].element_size() * self.v_buffers[i].nelement()
            )
            used_bytes += (
                self.k_buffers[i].element_size() * self.batch_size * self.num_heads
                * self.current_seq_len * self.head_dim
                + self.v_buffers[i].element_size() * self.batch_size * self.num_heads
                * self.current_seq_len * self.head_dim
            )

        return {
            "total_allocated_mb": allocated_bytes / (1024 ** 2),
            "total_used_mb": used_bytes / (1024 ** 2),
            "utilization_percent": (
                used_bytes / allocated_bytes * 100.0 if allocated_bytes > 0 else 0.0
            ),
            "seq_len": self.current_seq_len,
            "max_seq_len": self.max_seq_len,
            "num_layers": self.num_layers,
            "num_reallocations": self.stats.num_reallocations,
            "peak_allocated_mb": self.stats.peak_mb(),
            "wasted_mb": self.stats.wasted_mb(),
            "growth_bytes_per_step": self.stats.growth_rate_bytes_per_step(),
        }
