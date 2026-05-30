"""TTFT scaling sweep + chrome-trace profiles.

Two modes:
  --mode=timing   Run every (length, prompt, trial) combination, save CSV with
                  per-call TTFT. Used to fit the scaling curve and identify
                  where time goes as input grows.
  --mode=profile  Run ONE generate() per requested length wrapped in
                  vLLM's torch.profiler hook. Output: a chrome trace per
                  length, openable in chrome://tracing or perfetto.

Inputs:
  data/profile_prompts.json — produced by build_profile_prompts.py

Outputs (timing mode):
  runs/ttft_sweep_<tag>.csv         per-call TTFT
  runs/ttft_sweep_<tag>_summary.csv per-length median/min/p90/std
  runs/ttft_sweep_<tag>.json        config + GPU info

Outputs (profile mode):
  runs/profiles/profile_<tag>_len<N>/<rank>_<timestamp>.pt.trace.json.gz
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

from qwen_client import QwenClient

PROMPTS_PATH = Path("/data/users/jashwanth/qwen-claude/data/profile_prompts.json")
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, help="output filename suffix")
    p.add_argument("--mode", choices=["timing", "profile"], default="timing")
    p.add_argument("--lengths", default="",
                   help="comma-sep list of lengths. empty = all lengths in the file")
    p.add_argument("--prompts-per-length", type=int, default=5,
                   help="how many distinct prompts to use per length (timing mode)")
    p.add_argument("--trials-per-prompt", type=int, default=5,
                   help="generate() calls per (prompt, length); first is discarded as warmup")
    p.add_argument("--max-model-len", type=int, default=9216,
                   help="must be ≥ longest prompt + 1; default fits 8K prompts")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--cuda-graphs", action="store_true",
                   help="enable CUDA graphs (sets enforce_eager=False). Costs ~2 min more cold start.")
    p.add_argument("--prefix-cache", action="store_true",
                   help="enable vLLM prefix caching (default off for clean TTFT baselines)")
    return p.parse_args()


def load_prompts(lengths_filter: list[int] | None) -> dict[int, list[dict]]:
    """Return {length: [{prompt, prompt_id, actual_tokens}, ...]} for requested lengths."""
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"{PROMPTS_PATH} not found. Run build_profile_prompts.py first."
        )
    entries = json.loads(PROMPTS_PATH.read_text())
    by_length: dict[int, list[dict]] = {}
    for e in entries:
        if lengths_filter and e["length"] not in lengths_filter:
            continue
        by_length.setdefault(e["length"], []).append(e)
    return by_length


def run_timing(client: QwenClient, prompts_by_len: dict[int, list[dict]],
               prompts_per_length: int, trials_per_prompt: int) -> list[dict]:
    """Run the full (length, prompt, trial) grid. Returns per-call records."""
    records: list[dict] = []
    lengths = sorted(prompts_by_len.keys())

    # global warmup: longest available prompt × 3 calls to JIT all kernel paths
    longest = max(lengths)
    warm_prompt = prompts_by_len[longest][0]["prompt_token_ids"]
    print(f"\n[warmup] 3 calls at length {longest}…", flush=True)
    for _ in range(3):
        _ = client.generate(warm_prompt, max_tokens=1)

    for length in lengths:
        prompts = prompts_by_len[length][:prompts_per_length]
        print(f"\n[length={length}] {len(prompts)} prompts × {trials_per_prompt} trials"
              f"  ({len(prompts) * trials_per_prompt} calls)", flush=True)
        for prompt_idx, p in enumerate(prompts):
            for trial in range(trials_per_prompt):
                r = client.generate(p["prompt_token_ids"], max_tokens=1)
                is_warmup = (trial == 0)
                records.append({
                    "length": length,
                    "prompt_id": p["prompt_id"],
                    "actual_tokens": r.n_prompt_tokens,
                    "trial": trial,
                    "ttft_ms": r.ttft_ms,
                    "is_warmup": is_warmup,
                    "first_token": r.text[:20],
                })
            ttfts = [rec["ttft_ms"] for rec in records[-trials_per_prompt:]]
            warm_ttfts = ttfts[1:]  # drop trial 0
            print(f"  prompt {prompt_idx} (n={r.n_prompt_tokens}): "
                  f"all={[f'{t:.0f}' for t in ttfts]}  "
                  f"warm_median={statistics.median(warm_ttfts):.1f}ms",
                  flush=True)

    return records


def summarize(records: list[dict]) -> list[dict]:
    """Per-length stats over the warm (non-warmup) trials."""
    by_length: dict[int, list[float]] = {}
    for r in records:
        if r["is_warmup"]:
            continue
        by_length.setdefault(r["length"], []).append(r["ttft_ms"])

    summary = []
    for length in sorted(by_length):
        vals = by_length[length]
        summary.append({
            "length": length,
            "n": len(vals),
            "min_ms": min(vals),
            "p50_ms": statistics.median(vals),
            "p90_ms": statistics.quantiles(vals, n=10)[8] if len(vals) >= 10 else max(vals),
            "mean_ms": statistics.mean(vals),
            "std_ms": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
        })
    return summary


def write_timing_outputs(tag: str, records: list[dict], summary: list[dict],
                         args, gpu_name: str):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RUNS_DIR / f"ttft_sweep_{tag}.csv"
    with csv_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    summary_path = RUNS_DIR / f"ttft_sweep_{tag}_summary.csv"
    with summary_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    meta_path = RUNS_DIR / f"ttft_sweep_{tag}.json"
    meta_path.write_text(json.dumps({
        "tag": tag,
        "config": vars(args),
        "gpu": gpu_name,
        "n_records": len(records),
    }, indent=2, default=str))

    print(f"\n=== SUMMARY ({tag}) ===")
    print(f"{'length':>7}  {'n':>4}  {'min':>7}  {'p50':>7}  {'p90':>7}  {'mean':>7}  {'std':>6}")
    for s in summary:
        print(f"{s['length']:>7}  {s['n']:>4}  "
              f"{s['min_ms']:>6.1f}  {s['p50_ms']:>6.1f}  {s['p90_ms']:>6.1f}  "
              f"{s['mean_ms']:>6.1f}  {s['std_ms']:>5.1f}")
    print(f"\nwrote:\n  {csv_path}\n  {summary_path}\n  {meta_path}")


def run_profile(client: QwenClient, prompts_by_len: dict[int, list[dict]],
                tag: str, lengths: list[int]):
    """One profiled generate() per requested length. Trace files dumped to disk."""
    profile_root = RUNS_DIR / "profiles" / f"profile_{tag}"
    profile_root.mkdir(parents=True, exist_ok=True)

    # vLLM looks at VLLM_TORCH_PROFILER_DIR at LLM construction time, so it
    # must be set before QwenClient is built. We re-export it per length so each
    # profile dumps to its own subdir — but vLLM may ignore the late update; if
    # so, all traces land in the originally-set dir and we sort by mtime.
    for length in sorted(lengths):
        if length not in prompts_by_len or not prompts_by_len[length]:
            print(f"[skip] no prompt for length {length}", flush=True)
            continue
        out_dir = profile_root / f"len{length}"
        out_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(out_dir)

        p = prompts_by_len[length][0]
        print(f"\n[profile] length={length} → {out_dir}", flush=True)

        # warmup at this length first (kernel selection is shape-aware)
        for _ in range(2):
            _ = client.generate(p["prompt_token_ids"], max_tokens=1)

        # captured run
        client.start_profile()
        r = client.generate(p["prompt_token_ids"], max_tokens=1)
        client.stop_profile()

        # vLLM flushes the trace asynchronously after stop_profile; give it time
        time.sleep(2)
        print(f"  ttft={r.ttft_ms:.1f}ms  first_token={r.text[:20]!r}", flush=True)
        traces = list(out_dir.glob("*.json.gz")) + list(out_dir.glob("*.json"))
        print(f"  trace files: {[t.name for t in traces]}", flush=True)


def main():
    args = parse_args()
    lengths_filter = [int(s) for s in args.lengths.split(",")] if args.lengths else None
    prompts_by_len = load_prompts(lengths_filter)
    if not prompts_by_len:
        raise RuntimeError("no prompts loaded — check --lengths and that the JSON exists")

    # VLLM_TORCH_PROFILER_DIR must be set BEFORE LLM construction for profile mode
    if args.mode == "profile":
        os.environ.setdefault("VLLM_TORCH_PROFILER_DIR",
                              str(RUNS_DIR / "profiles" / f"profile_{args.tag}"))

    client = QwenClient(
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=args.prefix_cache,
        enforce_eager=not args.cuda_graphs,
        thinking=False,
    )

    import torch
    gpu_name = torch.cuda.get_device_name(0)

    if args.mode == "timing":
        records = run_timing(client, prompts_by_len,
                             args.prompts_per_length, args.trials_per_prompt)
        summary = summarize(records)
        write_timing_outputs(args.tag, records, summary, args, gpu_name)
    else:
        lengths = sorted(prompts_by_len.keys())
        run_profile(client, prompts_by_len, args.tag, lengths)


if __name__ == "__main__":
    main()
