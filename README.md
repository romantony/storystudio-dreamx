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

Checkpoint sizes (bf16, as loaded for inference) with this worker's default
of `GPU_MEMORY_MODE=model_full_load` plus explicit T5/VAE CPU offload flags
(set in `model_server.py`). Note that upstream's `model_cpu_offload` also
offloads the DiT itself, keeping it in host RAM and copying it to the GPU
every job — avoid it on Serverless workers with limited RAM.

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

T5 and the VAEs stay offloaded regardless of `GPU_MEMORY_MODE` (explicit
flags in `model_server.py`); only the DiT placement changes.

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

Download source: `GD-ML/DreamX-Creator` (Apache 2.0), ~55GB total, mirrored
identically (same paths, same byte sizes) on both
[Hugging Face](https://huggingface.co/GD-ML/DreamX-Creator) and
[ModelScope](https://modelscope.cn/models/GD-ML/DreamX-Creator) — use
whichever is faster from your pod's region (Hugging Face has been very slow
for this repo from some regions; ModelScope is the default below). Place
under `MODEL_PATH` (default `/runpod-volume/dreamx-creator`):

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
# ModelScope (default source)
pip install -U modelscope
python3 scripts/download_weights.py --dest /path/to/network/volume/dreamx-creator

# Hugging Face
pip install -U "huggingface_hub[cli]" hf_xet
python3 scripts/download_weights.py --source hf --dest /path/to/network/volume/dreamx-creator
```

See [`scripts/download_weights.py`](./scripts/download_weights.py) for
details — it verifies file counts per directory after download and is safe
to re-run (resumes rather than restarting). Pass `--include-refiner` if you
later add the 2K refiner stage and want its weights fetched too. Pass
`--clean` to wipe `--dest` first (e.g. to discard a stalled/partial download
before switching sources) — it prompts for confirmation unless `--yes` is
also given.

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
| `num_inference_steps` | int | ❌ | `DEFAULT_STEPS` (50) | Clamped to 10–100; below 50 is unverified quality (no distilled checkpoint exists) |
| `guidance_scale` | float | ❌ | 5.0 | Text CFG scale |
| `seed` | int | ❌ | random | Omit for a random seed per job; the seed used is returned in the response so a clip can be reproduced |
| `project_id` / `frame_id` | string | ❌ | - | Used for the R2 asset key when both given; falls back to timestamp+job_id |

### Response

```json
{
  "video_url": "https://storyaistudio.app/storystudio/video/20260912_abc123_dreamx_av.mp4",
  "generation_time": 420.0,
  "model_generation_time": 395.0,
  "video_size_mb": 8.1,
  "duration_s": 5.0,
  "num_inference_steps": 50,
  "seed": 1834920571
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
| `GPU_MEMORY_MODE` | `model_full_load` | `model_cpu_offload` also offloads the DiT to host RAM each job (see VRAM budget) |
| `VIDEOX_ATTENTION_TYPE` | `SAGE_ATTENTION` | SageAttention 2.2 (built for Ada 8.9 / Blackwell 12.0); set `FLASH_ATTENTION` to fall back to PyTorch SDPA |
| `TARGET_SPATIAL_TOKENS` | `880` | Repo default spatial-token budget for the resized first frame |
| `OUTPUT_FPS` | `24` | Repo default |
| `VIDEO_CRF` | `18` | libx264 quality for the output MP4 (lower = higher quality, larger file; ~23 is visibly softer) |
| `DEFAULT_STEPS` | `50` | Steps used when a request omits `num_inference_steps` |
| `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET_NAME` / `R2_PUBLIC_URL` | see `handler.py` | Cloudflare R2 upload target |

## 2K Refiner endpoint (`refiner/`)

A separate worker and image (`romantony/dreamx-creator-refiner`, built by
`.github/workflows/docker-build-refiner.yml` on changes under `refiner/`) for
DreamX's SR-DiT 2x video refiner (`video_refiner/` upstream). It takes a
finished clip, e.g. this worker's output, and returns a ~2x upscaled MP4 with
the source audio copied through. The model server runs upstream's
`inference_sr.py` unmodified: its setup once at startup, its inference loop per
job, with defaults matching `run_inference.sh`.

**Weights:** it uses the same network volume. Add `refiner/` (~11 GB) alongside
the existing base weights (base + refiner ≈ 51 GB):

```bash
python3 scripts/download_weights.py --dest /workspace/dreamx-creator --include-refiner
python3 scripts/check_download.py --dest /workspace/dreamx-creator --include-refiner
```

**Endpoint:** image `romantony/dreamx-creator-refiner:latest`, the same network
volume, the same R2 env vars, and an execution timeout of at least 1800s.
Upstream only benchmarks 1248x704 -> 2K on an H20 (96 GB). With upstream's
`kv_len=9`, a 48 GB card OOMs at 2496x1408 after 2 of 10 chunks: the rolling KV
cache dominates VRAM, so this worker defaults to `REFINER_KV_LEN=3`. Each job logs
`Peak VRAM x / y GB` — raise `REFINER_KV_LEN` while there is headroom (more
temporal context), or use a 96 GB RTX PRO 6000 for upstream's 9.

**API** (same shape as PostProd-Lite's `upscale` mode):

| Input | Default | |
|---|---|---|
| `video_url` | required | |
| `target_height` | - | Output height; returns the source unchanged (`upscale: "skipped_source_hires"`) if it is already that tall |
| `sr_scale` | `2.0` | Used when `target_height` is omitted; 1.0–2.25 |
| `seed` | `42` | |
| `project_id` / `frame_id` | - | R2 key naming, as in the base worker |

Output: `video` / `video_url`, `upscale` (`"dreamx_sr_<H>p"`), `width`,
`height`, `sr_scale`, `seed`, `video_size_mb`, `model_generation_time`,
`generation_time`. Test with `scripts/test_refiner.py`.

| Env var | Default | Purpose |
|---|---|---|
| `REFINER_FAST` | `0` | `1` = fp8 DiT + LightVAE-NU decoder (upstream: 155.8s -> 46.2s per clip, ~36 dB PSNR vs default) |
| `REFINER_KV_LEN` | `3` | Rolling KV cache length in 3-frame chunks; VRAM scales with it (upstream: 9) |
| `REFINER_WINDOW_CHUNK` | unset | Windows per batch in block-grid attention, to bound peak memory |
| `VIDEO_CRF` | `18` | libx264 quality of the refined MP4 |
| `MAX_INPUT_FRAMES` | `241` | Rejects longer inputs |

## Status

Dockerfile, handler, and model server are written and internally consistent
with `audio_video_generation/inference.py`'s actual function signatures
(`setup_models`, `generate_joint_audio_video`, `resolve_cpu_offload_flags`)
as of commit `215d4cd` of the upstream repo. **Not yet done:** downloading
weights to a real network volume, building/testing the Docker image, and
deploying against a live RunPod endpoint to measure actual VRAM peak and
per-clip latency — treat every number in this README under "Expected
performance" as a hypothesis to verify, not a confirmed spec.
