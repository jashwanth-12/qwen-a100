"""Measure TTFT on tau-bench multi-turn customer-service prompts.

Voice agent scenario: warm kernels, single request, greedy decode, thinking OFF.

Bucketing strategy: tau-bench split-turn prompts span ~1.4K-6K tokens depending on
how many history turns are loaded into a given row. We take the natural distribution
and bucket by token count to mirror voice-agent conversation-length variability.

Source: amityco/apigen-tau-bench-split-turn (HF mirror of Sierra τ-bench).
"""
import argparse
import glob
import json
import os
import random
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

MODEL_DIR = "/data/users/jashwanth/qwen-claude/models/Qwen3.6-35B-A3B-FP8"
DATA_GLOB = "/data/users/jashwanth/qwen-claude/data/taubench/taubench_apigen_*.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--buckets", default="1500-2000,2000-3000,3000-4000,4000-6000",
                   help="comma-sep min-max token ranges")
    p.add_argument("--prompts-per-bucket", type=int, default=8)
    p.add_argument("--trials-per-prompt", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--enable-prefix-caching", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-scan", type=int, default=4000,
                   help="only tokenize this many rows when collecting prompts")
    return p.parse_args()


def load_taubench_prompts(tok, buckets, per_bucket, seed, max_scan):
    """Return {bucket_label: [(prompt_text, n_tokens), ...]} from tau-bench."""
    paths = sorted(glob.glob(DATA_GLOB))
    print(f"loading tau-bench from {len(paths)} parquet shards…", flush=True)
    rows = []
    for p in paths:
        rows.extend(pq.read_table(p).to_pylist())
    print(f"  {len(rows)} total rows", flush=True)

    rng = random.Random(seed)
    rng.shuffle(rows)

    bucket_ranges = []
    for b in buckets:
        lo, hi = b.split("-")
        bucket_ranges.append((b, int(lo), int(hi)))

    out = {label: [] for label, _, _ in bucket_ranges}

    scanned = 0
    for row in rows:
        if all(len(v) >= per_bucket for v in out.values()):
            break
        if scanned >= max_scan:
            break
        scanned += 1
        msgs = row.get("messages") or []
        clean = [{"role": m["role"], "content": m["content"] or ""}
                 for m in msgs if m.get("role") in ("system", "user", "assistant")]
        # ensure last message is user (so we're prompting model for next assistant turn)
        while clean and clean[-1]["role"] == "assistant":
            clean.pop()
        if not clean or clean[-1]["role"] != "user":
            continue
        try:
            prompt = tok.apply_chat_template(
                clean, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            n = len(tok.encode(prompt))
        except Exception:
            continue
        for label, lo, hi in bucket_ranges:
            if lo <= n < hi and len(out[label]) < per_bucket:
                out[label].append((prompt, n))
                break
    print(f"  scanned {scanned} rows", flush=True)
    for label in out:
        print(f"  bucket {label}: {len(out[label])} prompts "
              f"(token counts: {[n for _, n in out[label]]})", flush=True)
    return out


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buckets = [b.strip() for b in args.buckets.split(",")]

    print(f"[{time.strftime('%H:%M:%S')}] loading vLLM…", flush=True)
    t0 = time.time()
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=MODEL_DIR,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        dtype="auto",
        tensor_parallel_size=1,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    prompts_by_bucket = load_taubench_prompts(
        tok, buckets, args.prompts_per_bucket, args.seed, args.max_scan,
    )

    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1)

    # one global warmup to JIT-compile kernels — use a longer prompt to warm up large-batch path
    print(f"\n[{time.strftime('%H:%M:%S')}] global warmup (5 calls)…", flush=True)
    all_prompts = [p for prompts in prompts_by_bucket.values() for p in prompts]
    if not all_prompts:
        print("ERROR: no prompts found in any bucket", flush=True)
        return
    # pick the longest prompt for warmup so kernels are sized for worst case
    warm_prompt = max(all_prompts, key=lambda x: x[1])[0]
    for _ in range(5):
        _ = llm.generate([warm_prompt], sp, use_tqdm=False)

    results = {}
    for label in buckets:
        prompts = prompts_by_bucket[label]
        if not prompts:
            print(f"\n[skip] bucket {label} — no prompts", flush=True)
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] bucket {label} "
              f"({len(prompts)} prompts × {args.trials_per_prompt} trials)", flush=True)
        ttfts = []
        first_outputs = []
        for pi, (prompt, n) in enumerate(prompts):
            for ti in range(args.trials_per_prompt):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = llm.generate([prompt], sp, use_tqdm=False)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                ttfts.append(dt)
                if ti == 0 and pi < 2:
                    first_outputs.append((n, out[0].outputs[0].text))
            print(f"  prompt {pi+1}/{len(prompts)} (n={n:5d})  "
                  f"trials: {[f'{ttfts[-args.trials_per_prompt+i]:.1f}' for i in range(args.trials_per_prompt)]}",
                  flush=True)
        warm = ttfts
        results[label] = {
            "n_samples": len(ttfts),
            "all_ms": ttfts,
            "min": min(warm),
            "p50": statistics.median(warm),
            "p90": statistics.quantiles(warm, n=10)[8] if len(warm) >= 10 else max(warm),
            "p99": statistics.quantiles(warm, n=100)[98] if len(warm) >= 100 else max(warm),
            "mean": statistics.mean(warm),
            "sample_outputs": first_outputs,
        }

    out_path = OUT_DIR / f"ttft_{args.tag}.json"
    out_path.write_text(json.dumps({
        "tag": args.tag,
        "config": vars(args),
        "gpu": torch.cuda.get_device_name(0),
        "results": results,
    }, indent=2, default=str))
    print(f"\n=== SUMMARY ({args.tag}) ===", flush=True)
    print(f"{'bucket':>12}  {'n':>4}  {'min':>8}  {'p50':>8}  {'p90':>8}  {'p99':>8}  {'mean':>8}", flush=True)
    for label, r in results.items():
        print(f"{label:>12}  {r['n_samples']:>4}  "
              f"{r['min']:>7.1f}ms  {r['p50']:>7.1f}ms  "
              f"{r['p90']:>7.1f}ms  {r['p99']:>7.1f}ms  {r['mean']:>7.1f}ms", flush=True)
    print(f"\nfirst sample outputs (sanity):", flush=True)
    for label, r in results.items():
        for n, txt in r["sample_outputs"]:
            print(f"  [{label}] n={n}: {txt!r}", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
