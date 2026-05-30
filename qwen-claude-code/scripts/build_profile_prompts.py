"""Build the test set for TTFT profiling across a wide length range.

Default length list spans 1 → 8000:
  step 20: 1, 20, 40, ..., 200   (10 short — measure system-overhead floor)
  step 50: 250, 300, ..., 1000   (16 mid  — capture memory→compute crossover)
  step 100: 1100, ..., 8000      (70 long — compute-bound scaling)

Strategy: random token IDs per length. Two reasons this is the right call here:
  1. tau-bench has only a handful of distinct prompts ≥5K tokens; we need 8K
  2. random IDs saturate MoE expert dispatch (all 256 experts active per call),
     which is the worst-case memory-traffic scenario and matches the roofline
     we want to compare against. The B200 reference used this approach too.

Output: data/profile_prompts.json — list of:
  {"length": L, "prompt_id": i, "prompt_token_ids": [...], "actual_tokens": L}

Reproducible: fixed RNG seed so the same prompt IDs come out across runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_PATH = Path("/data/users/jashwanth/qwen-claude/data/profile_prompts.json")

# Length grid: dense near the floor, coarse in the compute-bound region
LENGTHS: list[int] = (
    [1] + list(range(20, 201, 20))
    + list(range(250, 1001, 50))
    + list(range(1100, 8001, 100))
)

N_PROMPTS_PER_LENGTH = 3   # 3 random prompts × 5 trials = 15 measurements/length
VOCAB_LO, VOCAB_HI = 1000, 150000  # avoid special-token range at the bottom
SEED = 42


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"building prompts for {len(LENGTHS)} lengths "
          f"({min(LENGTHS)} → {max(LENGTHS)})…", flush=True)
    print(f"  prompts per length: {N_PROMPTS_PER_LENGTH}")
    print(f"  token id range: [{VOCAB_LO}, {VOCAB_HI})")
    print(f"  total entries: {len(LENGTHS) * N_PROMPTS_PER_LENGTH}")

    entries = []
    for length in LENGTHS:
        for prompt_id in range(N_PROMPTS_PER_LENGTH):
            ids = [rng.randint(VOCAB_LO, VOCAB_HI - 1) for _ in range(length)]
            entries.append({
                "length": length,
                "prompt_id": prompt_id,
                "prompt_token_ids": ids,
                "actual_tokens": length,
            })

    OUT_PATH.write_text(json.dumps(entries))
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nwrote {len(entries)} entries to {OUT_PATH} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
