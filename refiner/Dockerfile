# DreamX-Creator 2K refiner (SR-DiT 5B, causal few-step 2x video SR) — RunPod Serverless.
# Shares the base worker's network volume layout: needs refiner/ plus
# wan2.2_ti2v_5b/ (VAE, UMT5-xxl, tokenizer) under /runpod-volume/dreamx-creator.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# build-essential + python3.12-dev: Triton JIT-compiles the block-grid window
# attention kernels at runtime and needs a C compiler and Python headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    build-essential ffmpeg git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed -U pip

WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip cache purge

ENV DREAMX_COMMIT=215d4cd7fbed7e161ab508ae1f85a8fee0536f62
RUN git clone https://github.com/AMAP-ML/DreamX-Creator.git /workspace/dreamx-creator && \
    cd /workspace/dreamx-creator && git checkout ${DREAMX_COMMIT}

# The refiner's model code calls flash_attention() directly, which asserts
# flash-attn is installed (despite requirements.txt listing it as optional).
# Official prebuilt wheel matching torch 2.10 / CUDA 12 / cp312 / cxx11 ABI.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r /workspace/dreamx-creator/video_refiner/requirements.txt \
    av==12.0.0 runpod==1.12.0 boto3 requests \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" && \
    python3 -m pip cache purge

RUN cd /workspace/dreamx-creator/video_refiner && DISABLE_FLEX_COMPILE=1 python3 -c "\
import torch, flash_attn, triton, runpod; \
from torchvision.io import read_video; \
assert torch._C._GLIBCXX_USE_CXX11_ABI, 'flash-attn wheel needs the cxx11 ABI torch build'; \
import pipeline_sr.causal_inference, wan.modules.sr_dit.models, utils.sr_dit_wrapper; \
print(f'OK — torch={torch.__version__} flash_attn={flash_attn.__version__} triton={triton.__version__} runpod={runpod.__version__}')"

# Upstream resolves weights via ../checkpoints/{refiner,wan2.2_ti2v_5b}; point that
# at the network volume (dangling at build time, valid once the volume mounts).
RUN rm -rf /workspace/dreamx-creator/checkpoints && \
    ln -s /runpod-volume/dreamx-creator /workspace/dreamx-creator/checkpoints

COPY handler.py /workspace/handler.py
COPY model_server.py /workspace/model_server.py

CMD ["python3", "-u", "/workspace/handler.py"]
