"""Interactive sandbox for poking at the model + bench utilities.

Open this file in VSCode. Each `# %%` block is a runnable cell — put your
cursor inside a cell and press Shift+Enter to send it to the Interactive
Window. Variables persist across cells.
"""
from __future__ import annotations

from pathlib import Path

# %% ── cell 1: paths and constants ─────────────────────────────────────
MODEL_DIR: Path = Path("/data/users/jashwanth/qwen-claude/models/Qwen3.6-35B-A3B-FP8")
DATA_DIR: Path = Path("/data/users/jashwanth/qwen-claude/data/taubench")
RUNS_DIR: Path = Path("/data/users/jashwanth/qwen-claude/runs")

print(f"model exists: {MODEL_DIR.exists()}")
print(f"shards:       {len(list(DATA_DIR.glob('*.parquet')))}")
print(f"runs:         {len(list(RUNS_DIR.glob('ttft_*.json')))}")


# %% ── cell 2: load tokenizer (fast, no GPU) ───────────────────────────
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
print(f"vocab size: {tok.vocab_size}")
print(f"chat template head: {tok.chat_template[:200] if tok.chat_template else None!r}")


# %% ── cell 3: see what thinking-on vs thinking-off prompts look like ──
def render(question: str, thinking: bool) -> str:
    """Render a single-turn prompt through the Qwen3 chat template."""
    msgs: list[dict[str, str]] = [{"role": "user", "content": question}]
    return tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


q = "What's the capital of France?"
print("--- thinking ON ---")
print(repr(render(q, thinking=True)[-200:]))
print("--- thinking OFF ---")
print(repr(render(q, thinking=False)[-200:]))


# %% ── cell 4: load a real tau-bench prompt ────────────────────────────
import pyarrow.parquet as pq  # noqa: E402

shard = pq.read_table(str(DATA_DIR / "taubench_apigen_0.parquet"))
print(f"rows: {shard.num_rows}, cols: {shard.column_names}")
row0 = shard.slice(0, 1).to_pydict()
print(f"first row 'messages' length: {len(row0['messages'][0])}")


# %% ── cell 5: read a TTFT run result ──────────────────────────────────
import json  # noqa: E402

run = json.loads((RUNS_DIR / "ttft_p1_taubench.json").read_text())
print(f"tag: {run['tag']}")
print(f"gpu: {run['gpu']}")
for bucket, r in run["results"].items():
    print(f"  {bucket}: p50={r['p50']:.1f}ms  p90={r['p90']:.1f}ms")


# %% ── cell 6: spin up vLLM (slow — ~90s; do once per session) ─────────
# Uncomment to load the model. Once loaded `llm` persists across cells.
#
# from vllm import LLM, SamplingParams  # noqa: E402
#
# llm = LLM(
#     model=str(MODEL_DIR),
#     max_model_len=4096,
#     gpu_memory_utilization=0.85,
#     enable_prefix_caching=False,
#     enforce_eager=True,
#     dtype="auto",
#     tensor_parallel_size=1,
#     trust_remote_code=True,
#     limit_mm_per_prompt={"image": 0, "video": 0},
# )
# sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
#
# out = llm.generate([render("Say hello in 5 words.", thinking=False)], sp)
# print(out[0].outputs[0].text)
