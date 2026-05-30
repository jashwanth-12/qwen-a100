# qwen-claude — TTFT optimization plan

**Goal:** TTFT < 150 ms for `Qwen3.6-35B-A3B-FP8` in a voice-agent scenario.
**Hardware:** 1× A100 80GB SXM (no native FP8 tensor cores — FP8 weights dequantized via Marlin/Machete to BF16).
**Use case:** between STT and TTS with partial transcripts; warm kernels assumed.
**TTFT definition:** wall-clock from `generate()` call to first token returned, kernels pre-warmed.

## Model

- **Arch:** `Qwen3_5MoeForConditionalGeneration` — 35B total / 3B activated.
- 40 hybrid layers: pattern `10 × (3 × GatedDeltaNet→MoE + 1 × GatedAttn→MoE)`.
  - 30 linear-attention layers (SSM-like, **no KV cache**, O(N) prefill).
  - 10 full-attention layers (head_dim=256, 16 Q heads, 2 KV heads → tiny KV).
- 256-expert MoE, top-8 routed + 1 shared expert (`moe_intermediate=512`).
- MTP head: 1 hidden layer (self-speculation — **not useful for TTFT**, only tokens 2+).
- Vision tower: present but drop with `--language-model-only` for voice.
- Quant: FP8 e4m3 block-wise `[128, 128]`, dynamic activation scaling.
- Weights: 37.46 GB across 42 shards.

## Serving stack

- `vllm >= 0.19.0` and `sglang >= 0.5.10` both natively support this arch.
- `transformers >= 4.57.1` required for HF path.
- Stick with vLLM as the primary path (single-stack, less fragile than HF).

## Phases

| # | Knob | What we measure | What we expect |
|---|---|---|---|
| **P0** | Setup | Download done, vLLM loads | Just works |
| **P1** | vLLM default, no prefix cache, `enforce_eager` | TTFT baseline at N ∈ {256, 1024, 2048} | Likely 200-400 ms — too slow |
| **P2** | + CUDA graphs, FA backend tune, chunked prefill, `--language-model-only` | TTFT delta vs P1 | -10-30% |
| **P3** | + `--enable-prefix-caching`, voice pattern (static prefix + delta) | TTFT for delta-only prefill | **The big win** — likely 50-100 ms |
| **P4** | Targeted profiling | MoE routing, attn kernel, Python overhead breakdown | Find the next bottleneck |
| **P5** | Stretch: BF16 dequant offline (Marlin overhead → 0) | TTFT vs P3 | -10-30% on Ampere; may regress on Hopper |

## Files

- `scripts/download_weights.py` — HF snapshot download
- `scripts/bench_ttft.py` — TTFT measurement harness (one tag per phase config)
- `runs/ttft_<tag>.json` — raw measurements per phase
- `models/Qwen3.6-35B-A3B-FP8/` — weights
- `logs/` — long-running script logs

## What we are NOT optimizing

- **TPOT** (tokens per second after first). MTP helps here, prefix cache doesn't.
- **Throughput** / batch size. Voice is single-stream.
- **Long context.** Voice prompts cap at ~2K tokens; the 262K context is irrelevant.
