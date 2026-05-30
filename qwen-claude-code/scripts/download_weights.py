"""Download Qwen3.6-35B-A3B-FP8 from HF into models/Qwen3.6-35B-A3B-FP8/.

37.46 GB across 42 shards. Uses snapshot_download with resume; falls back from
hf_transfer if it errors. Skips nothing — vLLM with --language-model-only just
disables the vision path at runtime, doesn't change file requirements.
"""
import os
import sys
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
DEST = "/data/users/jashwanth/qwen-claude/models/Qwen3.6-35B-A3B-FP8"

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

path = snapshot_download(
    repo_id=REPO,
    local_dir=DEST,
    max_workers=8,
    tqdm_class=None,  # plain progress
)
print(f"\nDONE: {path}", file=sys.stderr)
