"""
Block 1 — Profiling Harness

Runs three benchmark experiments on the baseline engine:
  1. Prefill scaling: prompt length vs prefill latency
  2. Decode scaling: generation length vs per-token latency
  3. KV memory growth: tracks VRAM as sequence grows

Results are saved to profiling/results_v0.json for comparison
against future blocks.

Usage:
    python profiling/benchmark.py --model Qwen/Qwen2-0.5B
    python profiling/benchmark.py --model Qwen/Qwen2-0.5B --device cpu
"""

import json
import time
import argparse
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.engine import BaselineInferenceEngine


# ---------------------------------------------------------------------------
# Benchmark experiments
# ---------------------------------------------------------------------------

def bench_prefill_scaling(engine, prompt_lengths: list[int]) -> list[dict]:
    """Experiment 1: how does prefill latency scale with prompt length?"""
    results = []
    base_prompt = "The history of artificial intelligence is " * 50  # ~10 tok/repeat

    for target_len in prompt_lengths:
        # Build a prompt of approximately target_len tokens
        words = base_prompt.split()
        prompt = ""
        while True:
            toks = engine.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            if toks >= target_len:
                break
            prompt += " " + words[toks % len(words)]

        actual_len = engine.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        trace = engine.generate(prompt, max_new_tokens=1)

        results.append({
            "experiment": "prefill_scaling",
            "target_prompt_tokens": target_len,
            "actual_prompt_tokens": actual_len,
            "prefill_ms": round(trace.prefill_latency_ms, 2),
        })
        print(f"  prefill @ {actual_len} tokens: {trace.prefill_latency_ms:.1f} ms")

    return results


def bench_decode_scaling(engine, gen_lengths: list[int]) -> list[dict]:
    """Experiment 2: how does per-token latency change as context grows?"""
    results = []
    prompt = "Explain the differences between Python and C++ in detail."

    for max_tokens in gen_lengths:
        trace = engine.generate(prompt, max_new_tokens=max_tokens, greedy=True)
        summary = trace.summary()

        # Record per-step latency evolution (first 10, mid, last 10)
        steps = trace.decode_steps
        sample_latencies = []
        if steps:
            sample_latencies = [s.latency_ms for s in steps[:10]]
            mid = len(steps) // 2
            sample_latencies += [s.latency_ms for s in steps[mid:mid+10]]
            sample_latencies += [s.latency_ms for s in steps[-10:]]

        results.append({
            "experiment": "decode_scaling",
            "max_tokens": max_tokens,
            "tokens_generated": summary["tokens_generated"],
            "tokens_per_sec": summary["tokens_per_sec"],
            "p50_ms": summary["latency_p50_ms"],
            "p95_ms": summary["latency_p95_ms"],
            "kv_peak_mb": summary["kv_peak_mb"],
            "sample_step_latencies_ms": [round(l, 2) for l in sample_latencies],
        })
        print(f"  decode @ {max_tokens} tokens: "
              f"{summary['tokens_per_sec']:.1f} tok/s  "
              f"p50={summary['latency_p50_ms']:.1f}ms  "
              f"p95={summary['latency_p95_ms']:.1f}ms")

    return results


def bench_kv_growth(engine, max_tokens: int = 200) -> list[dict]:
    """Experiment 3: track KV cache memory byte-by-byte as sequence grows."""
    prompt = "Once upon a time in a land far away, "
    trace = engine.generate(prompt, max_new_tokens=max_tokens, greedy=True)

    results = []
    for step in trace.decode_steps:
        results.append({
            "experiment": "kv_growth",
            "step": step.step,
            "kv_bytes": step.kv_bytes,
            "kv_mb": round(step.kv_bytes / (1024**2), 4),
        })

    kv_mem = engine.decode_text(trace)
    print(f"  KV growth over {len(trace.decode_steps)} steps: "
          f"peak={max(s.kv_bytes for s in trace.decode_steps)/(1024**2):.2f} MB")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block 1 profiling harness")
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B", help="HuggingFace model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="profiling/results_v0.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n[Block1 Benchmark] device={device}  model={args.model}\n")

    engine = BaselineInferenceEngine(args.model, device)

    all_results = {
        "version": "V0_baseline",
        "model": args.model,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiments": [],
    }

    print("\n--- Experiment 1: Prefill scaling ---")
    all_results["experiments"] += bench_prefill_scaling(
        engine, prompt_lengths=[32, 64, 128, 256, 512]
    )

    print("\n--- Experiment 2: Decode scaling ---")
    all_results["experiments"] += bench_decode_scaling(
        engine, gen_lengths=[32, 64, 128, 256]
    )

    print("\n--- Experiment 3: KV cache growth ---")
    all_results["experiments"] += bench_kv_growth(engine, max_tokens=200)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Block1] Results saved to {out_path}")

    # Print summary table
    print("\n=== Block 1 Baseline Summary ===")
    print(f"{'Experiment':<30} {'Key metric':<40}")
    print("-" * 70)
    for r in all_results["experiments"]:
        if r["experiment"] == "prefill_scaling":
            print(f"  prefill @ {r['actual_prompt_tokens']:>4} tok  →  {r['prefill_ms']:>8.1f} ms")
        elif r["experiment"] == "decode_scaling":
            print(f"  decode @ {r['max_tokens']:>4} tok   →  "
                  f"{r['tokens_per_sec']:>6.1f} tok/s  p95={r['p95_ms']:>6.1f} ms")


if __name__ == "__main__":
    main()