"""
Block 1 — Smoke test
Run this first to verify the engine works before the full benchmark.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --device cpu   # no GPU
"""

import argparse
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.engine import BaselineInferenceEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tokens", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n[Smoke Test] model={args.model}  device={device}\n")

    engine = BaselineInferenceEngine(args.model, device)

    prompts = [
        "The Jetson Orin is a GPU designed for",
        "In machine learning, the attention mechanism",
        "KV cache optimization reduces inference cost by",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt!r}")
        trace = engine.generate(prompt, max_new_tokens=args.max_tokens, greedy=True)
        text = engine.decode_text(trace)
        summary = trace.summary()

        print(f"  Generated: {text!r}")
        print(f"  Prefill:   {summary['prefill_ms']:.1f} ms")
        print(f"  Decode:    {summary['tokens_per_sec']:.1f} tok/s  "
              f"p50={summary['latency_p50_ms']:.1f}ms  "
              f"p95={summary['latency_p95_ms']:.1f}ms")
        print(f"  KV peak:   {summary['kv_peak_mb']:.3f} MB")

        # KV cache report
        kv = engine.model  # access through engine for now
        print(f"  Steps:     {len(trace.decode_steps)}")

    print("\n[Smoke Test] PASSED — engine is operational.")
    print("Run: python profiling/benchmark.py   for full experiments.")


if __name__ == "__main__":
    main()