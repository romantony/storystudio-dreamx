# DreamX-Creator 1.0 — Base Audio-Video Generator (RunPod Serverless)

Generate synchronized audio+video from a single image + text prompt using
[AMAP-ML/DreamX-Creator](https://github.com/AMAP-ML/DreamX-Creator)'s 7B
joint generator. **This worker does not include the 2K refiner stage** — it's
the base ~5s clip only (image resolution, native output, not upscaled).

## How this differs from the Wan2.2 Lightning worker

| Aspect | wan22-14B-fp8-4steps | This worker |
|---|---|---|
| Model | Wan2.2-I2V-A14B, LightX2V 4-step distilled FP8 | DreamX-Creator 7B, **not distilled** — 50-step default |
| Output | Video only | Video **+ native synced audio** (no separate TTS/BGM step) |
| VRAM (resident) | ~31 GB (both 14B DiTs on GPU, T5 CPU-offloaded) | ~14 GB (creator only; T5+VAEs CPU-offloaded via the repo's own flags) |
| Speed lever | 4-step LightX2V distillation (~3.5x vs 15-30 step FP8) | None yet — repo's roadmap lists a distilled/faster release as not-yet-shipped |
| Loader complexity | Custom FP8 per-block loaders, RoPE float64→float32 VRAM patch | None needed — `inference.py`'s `setup_models()`/`generate_joint_audio_video()` used directly |

**Known-slow, by design of the upstream release:** with no distilled
checkpoint, expect noticeably longer per-clip latency than the 4-step
Lightning worker. Benchmark on your own pod before committing this to
production traffic — see "Expected performance" below for the reasoning
behind the estimate and how to validate it.

## VRAM budget (RTX 6000 Ada, 47.4 GB)

Checkpoint sizes (bf16, as loaded for inference) with
`GPU_MEMORY_MODE=model_cpu_offload` (the default in this worker):

| Component | Resident VRAM |
|---|---|
| Creator video+audio DiT + cross-attn (7B) | ~14 GB |
| UMT5-xxl text encoder | 0 GB (CPU; staged to GPU only during encode) |
| Video + audio VAEs | 0 GB (CPU; staged to GPU only during encode/decode) |
| **Base resident** | **~14 GB** |
| Available for activations (50-step, CFG-tripled batch) | ~33 GB |

This has far more headroom than the Wan2.2 Lightning worker's ~16GB
(which needed a manual RoPE float64→float32 patch to survive on the same
card). No equivalent patch is expected to be necessary here, but **peak
activation memory during sampling is not published by the DreamX-Creator repo
or paper** — confirm empirically on first deploy, the same way the Wan2.2
worker's 113-frame cap was found by testing 5/6/7/8s clips until one OOM'd.

If warm-start per-job latency from repeatedly staging T5/VAE to GPU turns out
to dominate wall time, try `GPU_MEMORY_MODE=model_full_load` instead — full
resident load is ~29 GB, still comfortably under 47.4 GB with no offload
transfer overhead per job.

## Expected performance (unverified — benchmark before production use)

The DreamX-Creator repo publishes an official benchmark only for the **2K
refiner** (not included in this worker): 155.8s/clip on a single H20 at
bf16 defaults, down to 46.2s with fp8+LightVAE. **No official latency number
exists for the base 7B generator** covered by this worker.

Reasoning for a rough estimate: default is 50 diffusion steps, and DreamX's
multimodal CFG mode (`--cfg_mode multimodal`, the default) triples the
per-step batch (three model evaluations: unconditional, bridge, text+bridge —
see `DirectionalMultimodalCFGAdapter` in `inference.py`), making it roughly
3x heavier per step than a plain Wan2.2-TI2V-5B forward pass at the same step
count. Un-distilled Wan2.2-TI2V-5B itself typically takes several minutes per
5s clip on a high-end GPU. Expect this worker to land in a similar
multi-minute-per-clip range on an RTX 6000 Ada — **treat this as an
order-of-magnitude guess, not a spec, until measured on the actual pod.**

`num_inference_steps` is exposed in the API (see below) as a lever to trade
quality for speed, but going below 50 is unverified — the repo has not
released a checkpoint distilled for fewer steps (unlike the Wan2.2 Lightning
DiTs, which *were* trained for exactly 4 steps).

## Model Files (Network Volume)

Download source: [`GD-ML/DreamX-Creator`](https://huggingface.co/GD-ML/DreamX-Creator)
(Apache 2.0), ~55GB total. Place under `MODEL_PATH` (default
`/runpod-volume/dreamx-creator`):

```
dreamx-creator/
├── creator/
│   ├── video_model/            # config.json + safetensors shards
│   ├── audio_model/            # config.json + safetensors
│   └── cross_attn_weights.safetensors
├── audio_vae/                  # CreatorDACVAE
└── wan2.2_ti2v_5b/
    ├── Wan2.2_VAE.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    └── google/umt5-xxl/        # tokenizer files
```

This worker does **not** need `refiner/` (that's the 2K stage, out of scope
here) — skip it to save ~11GB of download if disk space on the volume is
tight.

**Mount-point gotcha (same as the Wan2.2 worker):** download on a RunPod
CPU/storage Pod with the network volume attached, and check `df -h` first —
interactive Pods often mount the volume at `/workspace`, while
`/runpod-volume` on that same Pod is local ephemeral disk. Only the
Serverless endpoint itself mounts the volume at `/runpod-volume`.

```bash
pip install -U "huggingface_hub[cli]"
python3 scripts/download_weights.py --dest /path/to/network/volume/dreamx-creator
```

See [`scripts/download_weights.py`](./scripts/download_weights.py) for
details — it verifies file counts per directory after download and is safe
to re-run (resumes rather than restarting). Pass `--include-refiner` if you
later add the 2K refiner stage and want its weights fetched too.

## API Usage

```bash
curl -X POST https://api.runpod.ai/v2/{endpoint_id}/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {api_key}" \
  -d '{
    "input": {
      "image": "https://example.com/image.png",
      "prompt": "A cat walking gracefully through the scene, birds chirping",
      "duration_s": 5,
      "num_inference_steps": 50
    }
  }'
```

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | ✅ | - | Image URL or base64 encoded — becomes the first video frame |
| `prompt` | string | ✅ | - | Describes both motion/action AND sound — this model generates audio jointly, so mention diegetic sound in the prompt |
| `negative_prompt` | string | ❌ | repo default | |
| `duration_s` | float | ❌ | 5.0 | Clamped to 2.0–8.0s; only 5.0 is officially validated (Verse-Bench default) |
| `num_inference_steps` | int | ❌ | 50 | Clamped to 10–100; below 50 is unverified quality (no distilled checkpoint exists) |
| `guidance_scale` | float | ❌ | 5.0 | Text CFG scale |
| `seed` | int | ❌ | 42 | |
| `project_id` / `frame_id` | string | ❌ | - | Used for the R2 asset key when both given; falls back to timestamp+job_id |

### Response

```json
{
  "video_url": "https://storyaistudio.app/storystudio/video/20260912_abc123_dreamx_av.mp4",
  "generation_time": 420.0,
  "model_generation_time": 395.0,
  "video_size_mb": 8.1,
  "duration_s": 5.0,
  "num_inference_steps": 50
}
```

Output is a single muxed MP4 (video track from the DiT, audio track from the
joint generator's audio stream) — no separate voice/BGM upload, since the
audio is generated in-line with the video, not as a controllable scripted
narration track. **This does not replace storystudio's existing voice/BGM
pipeline** for scripted dialogue — it's suited to ambient/non-dialogue clips
where diegetic sound synced to on-screen action is the goal.

## Docker Image

Build and push manually (no GitHub Actions workflow set up yet — add one
modeled on the Wan2.2 worker's `.github/workflows/docker-build.yml` once this
is validated):

```bash
docker build -t <dockerhub-user>/dreamx-creator-base:latest .
docker push <dockerhub-user>/dreamx-creator-base:latest
```

## RunPod Setup

1. Create a new Serverless Endpoint (separate from the Wan2.2 endpoints)
2. Use the image built above
3. GPU: RTX 6000 Ada (47.4 GB)
4. Attach the network volume containing the files described above
5. Set timeout: at least 1800s given the unverified, likely multi-minute
   per-clip latency (see "Expected performance") — raise further once you've
   measured actual wall-clock time on the pod

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/runpod-volume/dreamx-creator` | Root dir for creator/audio_vae/wan2.2_ti2v_5b |
| `GPU_MEMORY_MODE` | `model_cpu_offload` | `model_full_load` to keep T5+VAEs resident instead (see VRAM budget) |
| `TARGET_SPATIAL_TOKENS` | `880` | Repo default spatial-token budget for the resized first frame |
| `OUTPUT_FPS` | `24` | Repo default |
| `DEFAULT_STEPS` | `50` | Repo default; per-job `num_inference_steps` overrides this |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET_NAME` / `R2_PUBLIC_URL` | see `handler.py` | Cloudflare R2 upload target |

## Status

Dockerfile, handler, and model server are written and internally consistent
with `audio_video_generation/inference.py`'s actual function signatures
(`setup_models`, `generate_joint_audio_video`, `resolve_cpu_offload_flags`)
as of commit `215d4cd` of the upstream repo. **Not yet done:** downloading
weights to a real network volume, building/testing the Docker image, and
deploying against a live RunPod endpoint to measure actual VRAM peak and
per-clip latency — treat every number in this README under "Expected
performance" as a hypothesis to verify, not a confirmed spec.
