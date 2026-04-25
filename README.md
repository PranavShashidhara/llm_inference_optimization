# Block 1 — Baseline Inference Engine (V0)

Control system for the LLM inference optimization engine.
Everything here is **intentionally unoptimized** — every number is a benchmark target.

## What this block establishes

| Component | Implementation | Known bottleneck |
|-----------|---------------|-----------------|
| Model | Qwen2-0.5B via HuggingFace | Python-level overhead |
| KV cache | `[batch, heads, seq_len, head_dim]` contiguous | `torch.cat` on every step |
| Attention | Full QKᵀ materialization | O(T²) memory |
| Decode loop | Python `for` loop, one token/step | No batching, no kernel fusion |
| Profiling | wall-clock timing + VRAM tracking | Post-hoc only (no feedback loop) |

## Directory structure

```
block1_baseline/
├── model/
│   ├── kv_cache.py      # Naive KV cache with memory stats
│   └── engine.py        # Autoregressive decode engine + InferenceTrace
├── profiling/
│   └── benchmark.py     # Three benchmark experiments
├── scripts/
│   └── smoke_test.py    # Quick sanity check
└── requirements.txt
```

## Quick start

```bash
# 1. Install deps (on Jetson Orin, torch is pre-installed via JetPack)
pip install -r requirements.txt

# 2. Smoke test (30 tokens, checks the engine works)
python scripts/smoke_test.py --device cuda

# 3. Full benchmark suite
python profiling/benchmark.py --device cuda

# Results saved to: profiling/results_v0.json
```

## Metrics produced

- `prefill_ms`: latency for the prompt forward pass
- `tokens_per_sec`: decode throughput
- `latency_p50_ms`, `latency_p95_ms`: per-token latency distribution
- `kv_peak_mb`: peak VRAM used by KV cache
- KV growth curve: bytes per decode step (linear growth expected)

## What to look for

After running the benchmark, you should see:
- **Prefill**: roughly quadratic growth with prompt length (O(T²) attention)
- **Decode**: latency should increase slightly as context grows (KV reads get heavier)
- **KV memory**: linear growth — each step appends ~`2 * num_layers * num_heads * head_dim * 2 bytes`

These numbers become the baselines that Blocks 2–4 will beat.

## What comes next

| Block | Target | Expected improvement |
|-------|--------|---------------------|
| Block 2 | KV cache memory layout | Faster decode, less VRAM |
| Block 3 | Triton FlashAttention kernel | O(T) memory, faster prefill |
| Block 4 | Kernel optimization | Higher GPU utilization |
| Block 5 | Dynamic batching | Higher throughput |
| Block 6 | Adaptive scheduler | Lower tail latency |