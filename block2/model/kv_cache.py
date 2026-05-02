"""
Block 1 — Naive KV Cache (control system)
Layout: [batch, num_heads, seq_len, head_dim]

Intentionally unoptimized:
  - contiguous tensor allocation
  - append-based updates (torch.cat every step)
  - no memory alignment
  - no warp-friendly access patterns

Every inefficiency here is a future benchmark target.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVCacheStats:
    """Tracks memory consumption for profiling."""
    steps: list = field(default_factory=list)
    bytes_used: list = field(default_factory=list)

    def record(self, step: int, k: torch.Tensor, v: torch.Tensor):
        self.steps.append(step)
        self.bytes_used.append(
            (k.element_size() * k.nelement() + v.element_size() * v.nelement())
        )

    def growth_rate_bytes_per_step(self) -> float:
        if len(self.bytes_used) < 2:
            return 0.0
        deltas = [b - a for a, b in zip(self.bytes_used, self.bytes_used[1:])]
        return sum(deltas) / len(deltas)

    def peak_mb(self) -> float:
        if not self.bytes_used:
            return 0.0
        return max(self.bytes_used) / (1024 ** 2)


class NaiveKVCache:
    """
    Naive KV cache using a simple list of (key, value) tensors per layer.

    Shape per layer entry:
        key:   [batch_size, num_heads, seq_len, head_dim]
        value: [batch_size, num_heads, seq_len, head_dim]

    On every decode step, new k/v vectors are appended via torch.cat,
    allocating fresh contiguous memory each time. This is the worst-case
    access pattern — no reuse, full copy every step.
    """

    def __init__(self, num_layers: int, device: torch.device):
        self.num_layers = num_layers
        self.device = device
        self.cache: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers
        self.stats = KVCacheStats()
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
        """
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (key, value)
        else:
            cached_k, cached_v = self.cache[layer_idx]
            # torch.cat forces a full memory copy every step — intentional bottleneck
            new_k = torch.cat([cached_k, key], dim=2)
            new_v = torch.cat([cached_v, value], dim=2)
            self.cache[layer_idx] = (new_k, new_v)

        # Record stats for layer 0 only (representative)
        if layer_idx == 0:
            k_full, v_full = self.cache[0]
            self.stats.record(self._step, k_full, v_full)
            self._step += 1

        return self.cache[layer_idx]

    def get(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return self.cache[layer_idx]

    def seq_len(self) -> int:
        if self.cache[0] is None:
            return 0
        return self.cache[0][0].shape[2]

    def clear(self):
        self.cache = [None] * self.num_layers
        self.stats = KVCacheStats()
        self._step = 0

    def memory_report(self) -> dict:
        total_bytes = 0
        for entry in self.cache:
            if entry is not None:
                k, v = entry
                total_bytes += k.element_size() * k.nelement()
                total_bytes += v.element_size() * v.nelement()
        return {
            "total_mb": total_bytes / (1024 ** 2),
            "seq_len": self.seq_len(),
            "num_layers": self.num_layers,
            "peak_mb": self.stats.peak_mb(),
            "growth_bytes_per_step": self.stats.growth_rate_bytes_per_step(),
        }