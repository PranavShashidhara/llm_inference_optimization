"""
Block 2 — Comparative Benchmark: Naive vs Optimized KV Cache

Runs identical workloads on both cache implementations and measures:
  1. Decode latency (tokens/sec)
  2. KV cache memory overhead
  3. Memory access efficiency
  4. Scaling behavior as sequence grows

Model: distilgpt2 (small, fast to iterate)

Usage:
    python profiling/benchmark_block2.py --model distilgpt2
    python profiling/benchmark_block2.py --model distilgpt2 --device cpu
"""

import json
import time
import argparse
import torch
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.engine_block2 import DualModeInferenceEngine


# ---------------------------------------------------------------------------
# Comparative experiments
# ---------------------------------------------------------------------------

def bench_decode_latency(
    engine_naive,
    engine_optimized,
    prompt: str = "The future of artificial intelligence is",
    gen_lengths: List[int] = None,
) -> dict:
    """
    Compare decode latency (tokens/sec) for same generation tasks.
    """
    if gen_lengths is None:
        gen_lengths = [32, 64, 128, 256]

    results = {
        "experiment": "decode_latency_comparison",
        "prompt": prompt,
        "comparisons": [],
    }

    for max_tokens in gen_lengths:
        print(f"\n  Comparing decode @ {max_tokens} tokens...")

        # Naive
        trace_naive = engine_naive.generate(prompt, max_new_tokens=max_tokens, greedy=True)
        summary_naive = trace_naive.summary()

        # Optimized
        trace_opt = engine_optimized.generate(prompt, max_new_tokens=max_tokens, greedy=True)
        summary_opt = trace_opt.summary()

        # Compute speedup
        tok_per_sec_naive = summary_naive["tokens_per_sec"]
        tok_per_sec_opt = summary_opt["tokens_per_sec"]
        speedup = tok_per_sec_opt / tok_per_sec_naive if tok_per_sec_naive > 0 else 1.0

        results["comparisons"].append({
            "max_tokens": max_tokens,
            "naive_tokens_per_sec": round(tok_per_sec_naive, 2),
            "optimized_tokens_per_sec": round(tok_per_sec_opt, 2),
            "speedup_percent": round((speedup - 1.0) * 100, 1),
            "naive_p95_ms": round(summary_naive["latency_p95_ms"], 2),
            "optimized_p95_ms": round(summary_opt["latency_p95_ms"], 2),
            "naive_kv_peak_mb": round(summary_naive["kv_peak_mb"], 3),
            "optimized_kv_peak_mb": round(summary_opt["kv_peak_mb"], 3),
        })

        print(f"    Naive:      {tok_per_sec_naive:>6.2f} tok/s  p95={summary_naive['latency_p95_ms']:>6.2f}ms")
        print(f"    Optimized:  {tok_per_sec_opt:>6.2f} tok/s  p95={summary_opt['latency_p95_ms']:>6.2f}ms  "
              f"→ {speedup:.2f}x")

    return results


def bench_kv_memory_efficiency(
    engine_naive,
    engine_optimized,
    prompt: str = "Once upon a time, ",
    max_tokens: int = 200,
) -> dict:
    """
    Compare memory efficiency: allocated vs used, fragmentation, reallocs.
    """
    print(f"\n  Benchmarking KV memory efficiency over {max_tokens} steps...")

    # Naive
    from model.kv_cache import NaiveKVCache
    kv_naive = NaiveKVCache(engine_naive.num_layers, engine_naive.device)
    trace_naive = engine_naive.generate(prompt, max_new_tokens=max_tokens, greedy=True)

    # Optimized
    from model.kv_cache_optimized import OptimizedKVCache
    kv_opt = OptimizedKVCache(
        num_layers=engine_optimized.num_layers,
        batch_size=1,
        num_heads=engine_optimized.num_heads,
        head_dim=engine_optimized.head_dim,
        device=engine_optimized.device,
        dtype=engine_optimized.dtype,
    )
    trace_opt = engine_optimized.generate(prompt, max_new_tokens=max_tokens, greedy=True)

    results = {
        "experiment": "kv_memory_efficiency",
        "max_tokens": max_tokens,
        "naive_stats": {
            "peak_mb": kv_naive.stats.peak_mb(),
            "growth_bytes_per_step": kv_naive.stats.growth_rate_bytes_per_step(),
        },
        "optimized_stats": {
            "peak_allocated_mb": kv_opt.stats.peak_mb(),
            "peak_used_mb": max(kv_opt.stats.bytes_used) / (1024**2) if kv_opt.stats.bytes_used else 0,
            "utilization_percent": kv_opt.stats.utilization_percent(),
            "wasted_mb": kv_opt.stats.wasted_mb(),
            "num_reallocations": kv_opt.stats.num_reallocations,
            "growth_bytes_per_step": kv_opt.stats.growth_rate_bytes_per_step(),
        },
    }

    print(f"    Naive peak:         {results['naive_stats']['peak_mb']:.3f} MB")
    print(f"    Optimized peak:     {results['optimized_stats']['peak_allocated_mb']:.3f} MB allocated, "
          f"{results['optimized_stats']['peak_used_mb']:.3f} MB used")
    print(f"    Utilization:        {results['optimized_stats']['utilization_percent']:.1f}%")
    print(f"    Wasted memory:      {results['optimized_stats']['wasted_mb']:.3f} MB")
    print(f"    Reallocations:      {results['optimized_stats']['num_reallocations']}")

    return results


def bench_latency_distribution(
    engine_naive,
    engine_optimized,
    prompt: str = "Hello, how are you?",
    max_tokens: int = 128,
) -> dict:
    """
    Compare latency distribution across decode steps.
    """
    print(f"\n  Comparing latency distribution over {max_tokens} steps...")

    # Naive
    trace_naive = engine_naive.generate(prompt, max_new_tokens=max_tokens, greedy=True)
    latencies_naive = [s.latency_ms for s in trace_naive.decode_steps]

    # Optimized
    trace_opt = engine_optimized.generate(prompt, max_new_tokens=max_tokens, greedy=True)
    latencies_opt = [s.latency_ms for s in trace_opt.decode_steps]

    # Compute stats
    def compute_stats(latencies):
        if not latencies:
            return {}
        sorted_lats = sorted(latencies)
        return {
            "mean_ms": round(sum(latencies) / len(latencies), 2),
            "p25_ms": round(sorted_lats[len(latencies) // 4], 2),
            "p50_ms": round(sorted_lats[len(latencies) // 2], 2),
            "p75_ms": round(sorted_lats[3 * len(latencies) // 4], 2),
            "p95_ms": round(sorted_lats[int(0.95 * len(latencies))], 2),
            "p99_ms": round(sorted_lats[int(0.99 * len(latencies))], 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
        }

    results = {
        "experiment": "latency_distribution",
        "max_tokens": max_tokens,
        "naive_stats": compute_stats(latencies_naive),
        "optimized_stats": compute_stats(latencies_opt),
    }

    print(f"    Naive       p50={results['naive_stats']['p50_ms']}ms  "
          f"p95={results['naive_stats']['p95_ms']}ms  "
          f"p99={results['naive_stats']['p99_ms']}ms")
    print(f"    Optimized   p50={results['optimized_stats']['p50_ms']}ms  "
          f"p95={results['optimized_stats']['p95_ms']}ms  "
          f"p99={results['optimized_stats']['p99_ms']}ms")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 2 comparative benchmark")
    parser.add_argument("--model", default="distilgpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="profiling/results_block2.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n[Block2 Benchmark] device={device}  model={args.model}\n")

    # Create two engines: naive and optimized
    print("Initializing naive (Block 1) engine...")
    engine_naive = DualModeInferenceEngine(args.model, device, cache_mode="naive")

    print("\nInitializing optimized (Block 2) engine...")
    engine_optimized = DualModeInferenceEngine(args.model, device, cache_mode="optimized")

    all_results = {
        "version": "V1_block2_comparative",
        "model": args.model,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiments": [],
    }

    print("\n" + "="*70)
    print("EXPERIMENT 1: Decode Latency Comparison")
    print("="*70)
    all_results["experiments"].append(
        bench_decode_latency(engine_naive, engine_optimized,
                            gen_lengths=[32, 64, 128, 256])
    )

    print("\n" + "="*70)
    print("EXPERIMENT 2: KV Memory Efficiency")
    print("="*70)
    all_results["experiments"].append(
        bench_kv_memory_efficiency(engine_naive, engine_optimized, max_tokens=200)
    )

    print("\n" + "="*70)
    print("EXPERIMENT 3: Latency Distribution")
    print("="*70)
    all_results["experiments"].append(
        bench_latency_distribution(engine_naive, engine_optimized, max_tokens=128)
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Block2] Results saved to {out_path}")

    # Print summary table
    print("\n" + "="*70)
    print("BLOCK 2 SUMMARY: Naive vs Optimized KV Cache")
    print("="*70)

    for exp in all_results["experiments"]:
        if exp["experiment"] == "decode_latency_comparison":
            print("\nDecode Latency (tokens/sec):")
            print(f"{'Tokens':<12} {'Naive':<15} {'Optimized':<15} {'Speedup':<12}")
            print("-" * 54)
            for comp in exp["comparisons"]:
                speedup_str = f"+{comp['speedup_percent']:.1f}%" if comp['speedup_percent'] > 0 else f"{comp['speedup_percent']:.1f}%"
                print(f"{comp['max_tokens']:<12} {comp['naive_tokens_per_sec']:<15.2f} "
                      f"{comp['optimized_tokens_per_sec']:<15.2f} {speedup_str:<12}")

        elif exp["experiment"] == "kv_memory_efficiency":
            print(f"\nKV Cache Memory (over {exp['max_tokens']} generation steps):")
            print(f"  Naive peak:           {exp['naive_stats']['peak_mb']:.3f} MB")
            print(f"  Optimized peak:       {exp['optimized_stats']['peak_allocated_mb']:.3f} MB (alloc), "
                  f"{exp['optimized_stats']['peak_used_mb']:.3f} MB (used)")
            print(f"  Utilization:          {exp['optimized_stats']['utilization_percent']:.1f}%")
            print(f"  Reallocations:        {exp['optimized_stats']['num_reallocations']}")


if __name__ == "__main__":
    main()
