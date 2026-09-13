# DreamX-Creator 1.0 base audio-video generator — RunPod Serverless Dockerfile
# Single 7B joint audio-video DiT (image+prompt -> synced video+audio), no
# refiner stage. Optimised for RTX 6000 Ada (47.4 GB, CUDA 12.8+).
#
# DreamX-Creator's own requirements.txt pins torch==2.10.0, which dropped
# Python 3.10 wheels — unlike the Wan2.2 Lightning worker's cu1290/torch260
# base image (Ubuntu 22.04 / py3.10), we need Python 3.12. Ubuntu 24.04 ships
# 3.12 by default, so we build from a plain CUDA devel image instead of a
# runpod/pytorch tag and install torch ourselves, exactly pinned to the cu128
# wheel (verified to exist) so the resulting driver requirement matches what
# RunPod's Ada hosts actually run — an unpinned index would otherwise resolve
# to a cu130 build that needs a newer driver than some hosts have.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/runpod-volume/dreamx-creator \
    HF_HOME=/runpod-volume/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    GPU_MEMORY_MODE=model_full_load

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    ffmpeg libsm6 libxext6 libglib2.0-0 git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# --ignore-installed: Ubuntu 24.04's apt-packaged python3-pip has no RECORD
# file, so pip's normal uninstall-before-upgrade step fails with "Cannot
# uninstall pip 24.0, RECORD file not found" — --ignore-installed skips that
# and just lays the new version on top.
RUN python3 -m pip install --no-cache-dir --break-system-packages --ignore-installed -U pip

WORKDIR /workspace

# Clone DreamX-Creator — provides audio_video_generation/{inference.py,videox_fun/,config/}.
# Pinned to a known-good commit rather than a floating branch tip.
ENV DREAMX_COMMIT=215d4cd7fbed7e161ab508ae1f85a8fee0536f62
RUN git clone https://github.com/AMAP-ML/DreamX-Creator.git /workspace/dreamx-creator && \
    cd /workspace/dreamx-creator && git checkout ${DREAMX_COMMIT}

# Pin torch/vision/audio to the exact versions the repo's requirements.txt
# specifies, but from the cu128 wheel index (not the default index, which
# would resolve a cu130 build) — same defensive pin as the Wan2.2 worker.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip cache purge

# flash-attn / sageattention are optional per the repo's own requirements.txt
# (PyTorch SDPA is the fallback); skip them for the initial build to keep it
# simple. Revisit if profiling shows attention is the bottleneck.

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r /workspace/dreamx-creator/audio_video_generation/requirements.txt \
    runpod==1.7.5 boto3==1.35.76 requests==2.32.3 && \
    python3 -m pip cache purge

RUN python3 -c "import runpod, torch, torchvision, diffusers, transformers; \
print(f'OK — runpod={runpod.__version__} torch={torch.__version__} cuda={torch.version.cuda}'); \
assert torch.version.cuda.startswith('12'), f'torch CUDA build {torch.version.cuda} requires too-new a driver'"

RUN test -f /workspace/dreamx-creator/audio_video_generation/inference.py && \
    test -d /workspace/dreamx-creator/audio_video_generation/videox_fun && \
    echo "DreamX-Creator source OK"

COPY handler.py /workspace/handler.py
COPY model_server.py /workspace/model_server.py

RUN mkdir -p /workspace/outputs

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3 -c "print('healthy')" || exit 1

CMD ["python3", "-u", "/workspace/handler.py"]
