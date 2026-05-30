"""Client for Qwen3.6-35B-A3B-FP8 served via vLLM.

Encapsulates model construction, tokenizer, chat-template rendering, and a
single generate() entry point. Benchmark and profile scripts import this so
they don't repeat the vLLM setup boilerplate.

Defaults match the voice-agent scenario:
  - greedy decode (temperature=0)
  - max_tokens=1 (TTFT measurement: prefill + first token)
  - thinking OFF (no <think>...</think> reasoning preamble)
  - prefix caching OFF, eager mode (clean per-call baseline; no graph reuse)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# absolute paths — code folder is separate from heavy folder (venv/models/data/runs)
MODEL_DIR = "/data/users/jashwanth/qwen-claude/models/Qwen3.6-35B-A3B-FP8"


@dataclass
class GenerateResult:
    """One generation, with the prompt that produced it and wall-clock timing."""
    text: str
    n_prompt_tokens: int
    n_output_tokens: int
    ttft_ms: float  # wall time prefill + first decode (with cuda sync)


class QwenClient:
    """vLLM-backed client for Qwen3.6-35B-A3B-FP8.

    Heavy: instantiating this loads ~37GB of FP8 weights + Marlin-dequantizes
    them to BF16. Takes ~90s. Hold one instance for the duration of your run.

    Light: each .generate() is just an LLM.generate() call, ~200-400ms at our
    sequence lengths.
    """

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        enable_prefix_caching: bool = False,
        enforce_eager: bool = True,
        tensor_parallel_size: int = 1,
        thinking: bool = False,
    ):
        # import inside __init__ so module-level import is cheap (lets scripts
        # import QwenClient just to call render_messages() without loading vLLM)
        from vllm import LLM
        from transformers import AutoTokenizer

        if not Path(model_dir).exists():
            raise FileNotFoundError(f"model dir not found: {model_dir}")

        self.model_dir = model_dir
        self.thinking = thinking

        print(f"[QwenClient] loading vLLM from {model_dir}…", flush=True)
        t0 = time.time()
        self.llm = LLM(
            model=model_dir,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            dtype="auto",
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print(f"[QwenClient] loaded in {time.time() - t0:.1f}s", flush=True)

    def render_messages(self, messages: list[dict[str, str]]) -> str:
        """Apply Qwen3 chat template. enable_thinking follows self.thinking.

        messages is OpenAI-style: [{"role": "system|user|assistant", "content": "..."}]
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(
        self,
        prompt: str | list[dict[str, str]] | list[int],
        max_tokens: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerateResult:
        """One generation. Accepts a rendered string, OpenAI-style messages, or
        a pre-tokenized list of int token IDs.

        Token-ID input is the canonical path for benchmarking — it gives exact
        sequence length control and avoids text→token round-trip drift.

        For TTFT measurement: default max_tokens=1, greedy, with cuda sync
        bookending the timing window. Returns text, token counts, and ttft_ms.
        """
        import torch
        from vllm import SamplingParams

        # build vLLM's prompt input — string OR TokensPrompt dict
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            # OpenAI-style messages → render via chat template
            vllm_input: Any = self.render_messages(prompt)  # type: ignore[arg-type]
        elif isinstance(prompt, list):
            # already token IDs
            vllm_input = {"prompt_token_ids": prompt}
        else:
            vllm_input = prompt

        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.llm.generate([vllm_input], sp, use_tqdm=False)
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1000

        o = out[0]
        return GenerateResult(
            text=o.outputs[0].text,
            n_prompt_tokens=len(o.prompt_token_ids),
            n_output_tokens=len(o.outputs[0].token_ids),
            ttft_ms=ttft_ms,
        )

    # ── profiling hooks (delegate to vLLM's built-in torch.profiler wrapper) ──
    # Set VLLM_TORCH_PROFILER_DIR=/path/before constructing the client. Then
    # start_profile() / stop_profile() around the calls you want to capture;
    # output is a chrome trace at the configured dir.
    def start_profile(self) -> None:
        self.llm.start_profile()

    def stop_profile(self) -> None:
        self.llm.stop_profile()
