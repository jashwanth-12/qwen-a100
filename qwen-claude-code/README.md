# qwen-claude

Inference experiments for `Qwen/Qwen3.6-35B-A3B-FP8` targeting **TTFT < 150 ms**
in a voice-agent scenario (between STT and TTS, warm kernels, single stream).

Hardware: 1× A100 80GB SXM.

See `PLAN.md` for the optimization phases.

## Layout

- `models/` — model weights (gitignored; downloaded via `scripts/download_weights.py`)
- `scripts/` — download + benchmarking scripts
- `runs/` — TTFT measurements as JSON, one per config tag
- `logs/` — long-running process logs
- `bench/` — ad-hoc bench artifacts

## Setup

```bash
source venv/bin/activate
export HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080

# Download weights (~37 GB)
python scripts/download_weights.py

# Install vLLM (>= 0.19.0 required for Qwen3.6 support)
pip install "vllm>=0.19.0" "transformers>=4.57.1" --torch-backend=auto
```

## Run a benchmark

```bash
# Phase 1 baseline
python scripts/bench_ttft.py --tag p1_baseline --enforce-eager

# Phase 2 with CUDA graphs
python scripts/bench_ttft.py --tag p2_cudagraphs

# Phase 3 prefix cache
python scripts/bench_ttft.py --tag p3_prefixcache --enable-prefix-caching
```

Output lands in `runs/ttft_<tag>.json`.
