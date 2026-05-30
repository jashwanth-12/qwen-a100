# source this before running any vllm script
# vllm 0.19.0 _C.abi3.so has unversioned cublas symbol references but no
# DT_NEEDED for libcublas — at dlopen time the system DCGM proxy
# (/lib64/libdcgm_cublas_proxy*) shadows the real libcublas symbols, so we
# LD_PRELOAD the venv's cu12 libcublas to win symbol resolution.

cd /data/users/jashwanth/qwen-claude
source venv/bin/activate

NVIDIA=/data/users/jashwanth/qwen-claude/venv/lib/python3.12/site-packages/nvidia
TORCH_LIB=/data/users/jashwanth/qwen-claude/venv/lib/python3.12/site-packages/torch/lib

export LD_LIBRARY_PATH="$NVIDIA/cublas/lib:$NVIDIA/cudnn/lib:$NVIDIA/cuda_runtime/lib:$NVIDIA/cuda_nvrtc/lib:$NVIDIA/cufft/lib:$NVIDIA/curand/lib:$NVIDIA/cusolver/lib:$NVIDIA/cusparse/lib:$NVIDIA/cusparselt/lib:$NVIDIA/nccl/lib:$NVIDIA/nvjitlink/lib:$NVIDIA/nvtx/lib:$NVIDIA/nvshmem/lib:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export LD_PRELOAD="$NVIDIA/cublas/lib/libcublasLt.so.12:$NVIDIA/cublas/lib/libcublas.so.12${LD_PRELOAD:+:$LD_PRELOAD}"

# vLLM defaults that suit our voice scenario
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1
