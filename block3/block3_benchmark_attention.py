"""
Block 3 — Attention Benchmark
Compares AttentionBaseline (full QKᵀ) vs TritonStreamingAttn (block-tiled).

Experiments:
    1. latency_scaling:   seq_len vs attention latency (ms)
    2. memory_scaling:    seq_len vs peak VRAM allocated during forward
    3. numerical_check:   max absolute error between baseline and streaming outputs

Usage:
    python block3_benchmark_attn.py
    python block3_benchmark_attn.py --device cpu --seq_max 512
    python block3_benchmark_attn.py --seq_max 2048 --repeats 20
"""

import argparse
import json
import time
import math
import torch
from pathlib import Path

from block3_attention import AttentionBaseline, TritonStreamingAttn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(batch, heads, seq_len, head_dim, device, dtype=torch.float16):
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def measure_latency(fn, *args, repeats=20, device_type="cuda", **kwargs):
    """Returns mean latency in ms."""
    # Warm-up
    for _ in range(3):
        _ = fn(*args, **kwargs)

    if device_type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(*args, **kwargs)
    if device_type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) * 1000.0 / repeats


def measure_peak_memory(fn, *args, device, **kwargs):
    """Returns peak VRAM delta in MB during fn call (CUDA only)."""
    if device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    baseline_mem = torch.cuda.memory_allocated(device)
    _ = fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    return (peak - baseline_mem) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Experiment 1 — Latency scaling
# ---------------------------------------------------------------------------

def bench_latency_scaling(
    device, batch=1, heads=16, head_dim=64,
    seq_lengths=(64, 128, 256, 512, 1024),
    repeats=20,
) -> list[dict]:
    print(f"\n[Exp 1] Latency scaling — seq_lengths={seq_lengths}")
    baseline = AttentionBaseline()
    streaming = TritonStreamingAttn()
    results = []

    for seq_len in seq_lengths:
        q, k, v = make_qkv(batch, heads, seq_len, head_dim, device)

        lat_b = measure_latency(
            baseline, q, k, v, causal=True,
            repeats=repeats, device_type=device.type
        )
        lat_s = measure_latency(
            streaming, q, k, v, causal=True,
            repeats=repeats, device_type=device.type
        )

        # Theoretical T² cost ratio (memory-bound prediction)
        theoretical_ratio = 1.0  # latency should scale the same; memory differs

        row = {
            "experiment": "latency_scaling",
            "seq_len": seq_len,
            "baseline_ms": round(lat_b, 3),
            "streaming_ms": round(lat_s, 3),
            "speedup": round(lat_b / max(lat_s, 0.001), 2),
        }
        results.append(row)
        print(
            f"  seq={seq_len:>5}  baseline={lat_b:.2f}ms  "
            f"streaming={lat_s:.2f}ms  speedup={row['speedup']}×"
        )

    return results


# ---------------------------------------------------------------------------
# Experiment 2 — Peak memory scaling
# ---------------------------------------------------------------------------

def bench_memory_scaling(
    device, batch=1, heads=16, head_dim=64,
    seq_lengths=(64, 128, 256, 512, 1024),
) -> list[dict]:
    print(f"\n[Exp 2] Memory scaling — seq_lengths={seq_lengths}")
    baseline = AttentionBaseline()
    streaming = TritonStreamingAttn()
    results = []

    for seq_len in seq_lengths:
        q, k, v = make_qkv(batch, heads, seq_len, head_dim, device)

        # Theoretical T² cost: batch × heads × T × T × dtype_bytes
        theoretical_scores_mb = (
            batch * heads * seq_len * seq_len * 2   # fp16 = 2 bytes
        ) / (1024 ** 2)

        mem_b = measure_peak_memory(baseline, q, k, v, causal=True, device=device)
        mem_s = measure_peak_memory(streaming, q, k, v, causal=True, device=device)

        row = {
            "experiment": "memory_scaling",
            "seq_len": seq_len,
            "baseline_peak_mb": round(mem_b, 3),
            "streaming_peak_mb": round(mem_s, 3),
            "theoretical_scores_mb": round(theoretical_scores_mb, 3),
            "memory_reduction": round(mem_b / max(mem_s, 0.001), 2),
        }
        results.append(row)
        print(
            f"  seq={seq_len:>5}  baseline={mem_b:.2f}MB  "
            f"streaming={mem_s:.2f}MB  "
            f"theoretical_scores={theoretical_scores_mb:.2f}MB  "
            f"reduction={row['memory_reduction']}×"
        )

    return results


# ---------------------------------------------------------------------------
# Experiment 3 — Numerical correctness
# ---------------------------------------------------------------------------

def bench_numerical_check(
    device, batch=1, heads=4, head_dim=64,
    seq_lengths=(32, 64, 128, 256),
) -> list[dict]:
    """
    Verify streaming attention output matches baseline (within fp16 tolerance).
    Uses fp32 inputs to isolate algorithmic error from dtype rounding.
    """
    print(f"\n[Exp 3] Numerical correctness check — seq_lengths={seq_lengths}")
    baseline = AttentionBaseline()
    streaming = TritonStreamingAttn()
    results = []

    for seq_len in seq_lengths:
        q, k, v = make_qkv(batch, heads, seq_len, head_dim, device, dtype=torch.float32)

        out_b = baseline(q, k, v, causal=True)
        out_s = streaming(q, k, v, causal=True)

        # Compute error
        abs_err = (out_b - out_s).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()
        passed = max_err < 1e-2   # fp16 tolerance

        row = {
            "experiment": "numerical_check",
            "seq_len": seq_len,
            "max_abs_error": round(max_err, 6),
            "mean_abs_error": round(mean_err, 8),
            "passed": passed,
        }
        results.append(row)
        status = "✓ PASS" if passed else "✗ FAIL"
        print(
            f"  seq={seq_len:>5}  max_err={max_err:.2e}  "
            f"mean_err={mean_err:.2e}  {status}"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 3 attention benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch",    type=int, default=1)
    parser.add_argument("--heads",    type=int, default=16)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_max",  type=int, default=1024,
                        help="Maximum sequence length to test")
    parser.add_argument("--repeats",  type=int, default=20)
    parser.add_argument("--output", default="profiling/results_block3.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n[Block 3 Attention Benchmark] device={device}  "
          f"batch={args.batch}  heads={args.heads}  dim={args.head_dim}\n")

    # Build seq length list up to seq_max
    seq_lengths = [s for s in [64, 128, 256, 512, 1024, 2048] if s <= args.seq_max]
    if not seq_lengths:
        seq_lengths = [args.seq_max]

    all_results = {
        "version": "V2_triton_attention",
        "device": str(device),
        "config": vars(args),
        "experiments": [],
    }

    lat_results = bench_latency_scaling(
        device, args.batch, args.heads, args.head_dim, seq_lengths, args.repeats
    )
    all_results["experiments"].extend(lat_results)

    mem_results = bench_memory_scaling(
        device, args.batch, args.heads, args.head_dim, seq_lengths
    )
    all_results["experiments"].extend(mem_results)

    num_results = bench_numerical_check(
        device, args.batch, min(4, args.heads), args.head_dim,
        [s for s in seq_lengths if s <= 256]
    )
    all_results["experiments"].extend(num_results)

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Block 3] Results saved → {out}")

    # Summary table
    print("\n=== Block 3 Attention Summary ===")
    print(f"{'seq_len':>8}  {'baseline_ms':>12}  {'streaming_ms':>13}  "
          f"{'speedup':>8}  {'mem_reduction':>14}")
    print("-" * 65)

    lat_by_seq = {r["seq_len"]: r for r in lat_results}
    mem_by_seq = {r["seq_len"]: r for r in mem_results}

    for seq_len in seq_lengths:
        lr = lat_by_seq.get(seq_len, {})
        mr = mem_by_seq.get(seq_len, {})
        print(
            f"{seq_len:>8}  "
            f"{lr.get('baseline_ms', 0):>11.2f}ms  "
            f"{lr.get('streaming_ms', 0):>12.2f}ms  "
            f"{lr.get('speedup', 0):>7.2f}×  "
            f"{mr.get('memory_reduction', 0):>13.2f}×"
        )

    # Check all numerical tests passed
    all_passed = all(r["passed"] for r in num_results)
    print(f"\nNumerical correctness: {'ALL PASSED ✓' if all_passed else 'FAILURES DETECTED ✗'}")


if __name__ == "__main__":
    main()