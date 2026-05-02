"""
Block 2 — Smoke Test
Validate that both naive and optimized KV cache implementations work correctly.

Checks:
  1. Numerical correctness (both should generate same tokens)
  2. Engine initialization (both cache modes)
  3. Generation works end-to-end
  4. Memory reporting (both implementations)

Usage:
    python scripts/smoke_test_block2.py
    python scripts/smoke_test_block2.py --device cpu
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")

import argparse
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.engine_block2 import DualModeInferenceEngine


def test_numerical_consistency(engine_naive, engine_optimized, prompt: str):
    """
    Both engines should generate identical token sequences (greedy mode).
    This validates that the optimized cache doesn't change model behavior.
    """
    print(f"\n[Test 1] Numerical Consistency")
    print(f"  Prompt: {prompt!r}")

    trace_naive = engine_naive.generate(prompt, max_new_tokens=50, greedy=True)
    trace_opt = engine_optimized.generate(prompt, max_new_tokens=50, greedy=True)

    tokens_naive = [s.token_id for s in trace_naive.decode_steps]
    tokens_opt = [s.token_id for s in trace_opt.decode_steps]

    if tokens_naive == tokens_opt:
        print(f"  ✓ PASS: Both engines generated identical sequences ({len(tokens_naive)} tokens)")
        return True
    else:
        print(f"  ✗ FAIL: Token sequences differ!")
        print(f"    Naive:     {tokens_naive[:10]}...")
        print(f"    Optimized: {tokens_opt[:10]}...")
        return False


def test_cache_modes(device: torch.device, model_name: str):
    """
    Test that both cache modes initialize and run.
    """
    print(f"\n[Test 2] Cache Modes Initialization")

    try:
        print(f"  Initializing naive cache mode...")
        engine_naive = DualModeInferenceEngine(model_name, device, cache_mode="naive")
        print(f"  ✓ Naive cache initialized")

        print(f"  Initializing optimized cache mode...")
        engine_opt = DualModeInferenceEngine(model_name, device, cache_mode="optimized")
        print(f"  ✓ Optimized cache initialized")

        return engine_naive, engine_opt
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        return None, None


def test_generation_end_to_end(engine, prompt: str, max_tokens: int = 30):
    """
    Test that generation works end-to-end without crashes.
    """
    print(f"\n[Test 3] Generation End-to-End")
    print(f"  Prompt: {prompt!r}")
    print(f"  Max tokens: {max_tokens}")

    try:
        trace = engine.generate(prompt, max_new_tokens=max_tokens, greedy=True)
        text = engine.decode_text(trace)
        summary = trace.summary()

        print(f"  Generated text: {text!r}")
        print(f"  Tokens generated: {summary['tokens_generated']}")
        print(f"  Prefill latency: {summary['prefill_ms']:.1f} ms")
        print(f"  Decode speed: {summary['tokens_per_sec']:.2f} tok/s")
        print(f"  KV peak memory: {summary['kv_peak_mb']:.3f} MB")
        print(f"  ✓ PASS: End-to-end generation successful")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_reporting(engine, prompt: str = "The quick brown fox"):
    """
    Test that memory reporting works for both cache implementations.
    """
    print(f"\n[Test 4] Memory Reporting")
    print(f"  Prompt: {prompt!r}")

    try:
        # Generate and get cache stats
        trace = engine.generate(prompt, max_new_tokens=50, greedy=True)
        
        # Access cache stats through trace
        summary = trace.summary()
        
        print(f"  KV cache peak: {summary['kv_peak_mb']:.3f} MB")
        print(f"  ✓ PASS: Memory reporting works")
        return True
    except Exception as e:
        print(f"  ✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Block 2 smoke test")
    parser.add_argument("--model", default="distilgpt2", help="HuggingFace model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_tokens", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*70}")
    print(f"[Block2 Smoke Test] model={args.model}  device={device}")
    print(f"{'='*70}")

    # Test 1: Initialize both cache modes
    engine_naive, engine_opt = test_cache_modes(device, args.model)
    if engine_naive is None or engine_opt is None:
        print("\n[Block2 Smoke Test] FAILED — could not initialize engines")
        return

    # Test 2: Numerical consistency
    prompts_test = [
        "The Jetson Orin is a GPU designed for",
        "In machine learning, the attention mechanism",
        "KV cache optimization reduces inference cost by",
    ]

    consistency_passed = 0
    for prompt in prompts_test:
        if test_numerical_consistency(engine_naive, engine_opt, prompt):
            consistency_passed += 1

    # Test 3: Generation end-to-end for each engine
    print(f"\n[Test 3] Generation End-to-End (Naive)")
    naive_ok = test_generation_end_to_end(engine_naive, prompts_test[0], args.max_tokens)

    print(f"\n[Test 3] Generation End-to-End (Optimized)")
    opt_ok = test_generation_end_to_end(engine_opt, prompts_test[0], args.max_tokens)

    # Test 4: Memory reporting
    print(f"\n[Test 4] Memory Reporting (Naive)")
    naive_mem_ok = test_memory_reporting(engine_naive)

    print(f"\n[Test 4] Memory Reporting (Optimized)")
    opt_mem_ok = test_memory_reporting(engine_opt)

    # Summary
    print(f"\n{'='*70}")
    print(f"SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Numerical consistency:  {consistency_passed}/{len(prompts_test)} passed")
    print(f"Naive generation:       {'✓ PASS' if naive_ok else '✗ FAIL'}")
    print(f"Optimized generation:   {'✓ PASS' if opt_ok else '✗ FAIL'}")
    print(f"Naive memory:           {'✓ PASS' if naive_mem_ok else '✗ FAIL'}")
    print(f"Optimized memory:       {'✓ PASS' if opt_mem_ok else '✗ FAIL'}")

    all_passed = (
        consistency_passed == len(prompts_test)
        and naive_ok and opt_ok and naive_mem_ok and opt_mem_ok
    )

    if all_passed:
        print(f"\n[Block2 Smoke Test] PASSED ✓")
        print(f"Ready to run: python profiling/benchmark_block2.py --model {args.model}")
    else:
        print(f"\n[Block2 Smoke Test] FAILED ✗")


if __name__ == "__main__":
    main()
