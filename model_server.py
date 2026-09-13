"""Persistent model server for DreamX-Creator's base audio-video generator.

Loads the 7B joint Creator generator once at container startup and serves
generation requests over a Unix socket, so the model stays warm between
RunPod jobs. Mirrors the ModelServer/socket pattern from
wan22-14B-fp8-4steps/model_server.py, but is much thinner: DreamX-Creator's
own audio_video_generation/inference.py already factors model loading
(setup_models) apart from per-item generation (generate_joint_audio_video),
so there is no need to re-implement custom checkpoint loaders the way the
Wan2.2 FP8 Lightning worker did.
"""
import os
import sys
import json
import socket
import time
import traceback
import subprocess
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = "/workspace/dreamx-creator/audio_video_generation"
sys.path.insert(0, REPO_ROOT)

MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/dreamx-creator")
SOCKET_PATH = "/tmp/dreamx_model_server.sock"

import torch  # noqa: E402
from torchvision.io import write_video  # noqa: E402

from inference import (  # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    init_device,
    resolve_weight_dtype,
    resolve_cpu_offload_flags,
    setup_models,
    generate_joint_audio_video,
    save_audio_wav,
)


def build_base_args() -> SimpleNamespace:
    """Static, job-independent config. Per-job fields (image, prompt,
    duration, seed, steps, output paths) are supplied via the `item` dict
    passed to generate_joint_audio_video, matching inference.py's own
    args-vs-item split — args stay fixed once the model is loaded."""
    return SimpleNamespace(
        config_path=os.path.join(REPO_ROOT, "config/config.yaml"),
        model_name=os.path.join(MODEL_PATH, "wan2.2_ti2v_5b"),
        transformer_path=os.path.join(MODEL_PATH, "creator"),
        audio_vae_path=os.path.join(MODEL_PATH, "audio_vae"),
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        # inference.py's generate_joint_audio_video() reads these off `args`
        # as item.get(key, args.<key>) fallback defaults — Python evaluates
        # that default eagerly, so args.duration/seed/image must exist even
        # though generate() below always supplies them in `item` too and
        # these values are never actually used.
        duration=5.0,
        seed=42,
        image=None,
        target_spatial_tokens=int(os.getenv("TARGET_SPATIAL_TOKENS", "880")),
        min_token_ratio=0.95,
        fps=int(os.getenv("OUTPUT_FPS", "24")),
        num_inference_steps=int(os.getenv("DEFAULT_STEPS", "50")),
        guidance_scale=5.0,
        cfg_mode="multimodal",
        video_bridge_guidance_scale=3.5,
        audio_bridge_guidance_scale=3.5,
        video_shift=5.0,
        audio_shift=5.0,
        sampler_name="Flow",
        weight_dtype="bfloat16",
        # model_cpu_offload keeps only the ~14GB creator DiT resident; T5 and
        # both VAEs stay on CPU except during their brief encode/decode phase.
        # On the RTX 6000 Ada (47.4GB) this leaves ample headroom for
        # activations even before considering model_full_load. Override to
        # model_full_load only if warm-start latency (T5/VAE re-transfer per
        # job) turns out to dominate wall time in practice.
        GPU_memory_mode=os.getenv("GPU_MEMORY_MODE", "model_cpu_offload"),
        text_encoder_cpu_offload=None,
        video_vae_cpu_offload=None,
        audio_vae_cpu_offload=None,
        vae_cpu_offload=None,
        use_temporal_rope=True,
        audio_fps=48000.0 / 960.0,
        vae_temporal_stride=4,
        disable_a2v_cross_attn=False,
        disable_v2a_cross_attn=False,
    )


class ModelServer:
    def __init__(self):
        self.device = None
        self.weight_dtype = None
        self.base_args = None
        self.models = None

    def load_model(self):
        print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
        try:
            import flash_attn
            print(f"FlashAttention available: v{flash_attn.__version__}")
        except ImportError:
            print("FlashAttention NOT installed — attention falls back to PyTorch SDPA")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"VRAM: {vram_gb:.1f} GB")

        for sub in ("creator", "audio_vae", "wan2.2_ti2v_5b"):
            p = Path(MODEL_PATH) / sub
            if not p.exists():
                raise RuntimeError(
                    f"Missing {sub}/ under {MODEL_PATH} — check the network volume "
                    f"is attached and populated per README.md's layout."
                )

        self.device = init_device()
        self.base_args = build_base_args()
        self.weight_dtype = resolve_weight_dtype(self.base_args.weight_dtype, self.device)

        offload_flags = resolve_cpu_offload_flags(self.base_args)
        print(
            "CPU offload: "
            f"text_encoder={offload_flags['text_encoder']} "
            f"video_vae={offload_flags['video_vae']} "
            f"audio_vae={offload_flags['audio_vae']}"
        )

        start = time.time()
        self.models = setup_models(self.base_args, self.device, self.weight_dtype)
        elapsed = time.time() - start
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"✓ Model ready in {elapsed:.1f}s — {allocated:.1f} GB alloc / {reserved:.1f} GB reserved")

    def generate(self, params: dict) -> dict:
        t0 = time.time()
        args = self.base_args
        args.output = params["output_path"]

        item = {
            "prompt": params["prompt"],
            "video_prompt": params["prompt"],
            "audio_prompt": params["prompt"],
            "negative_prompt": params.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT,
            "audio_negative_prompt": params.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT,
            "first_frame_path": params["image_path"],
            "duration": float(params.get("duration", 5.0)),
            "guidance_scale": float(params.get("guidance_scale", args.guidance_scale)),
            "num_inference_steps": int(params.get("num_inference_steps", args.num_inference_steps)),
            "seed": int(params.get("seed", 42)),
            "name": params.get("job_id", "job"),
        }

        video_decoded, audio_decoded, _ = generate_joint_audio_video(
            args, self.models, self.device, self.weight_dtype, item)

        output_path = params["output_path"]
        video_path = os.path.splitext(output_path)[0] + ".video.mp4"
        audio_path = os.path.splitext(output_path)[0] + ".wav"

        frames = (video_decoded[0].permute(1, 2, 3, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        write_video(video_path, torch.from_numpy(frames), fps=args.fps, video_codec="h264")
        save_audio_wav(audio_decoded, int(self.models["audio_vae"].sample_rate), audio_path)

        mux = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", output_path],
            check=False, capture_output=True, text=True,
        )
        if mux.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {mux.stderr.strip()}")

        torch.cuda.empty_cache()
        return {
            "success": True,
            "generation_time": time.time() - t0,
        }

    def run(self):
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(1)
        print(f"Model server listening on {SOCKET_PATH}", flush=True)

        while True:
            conn, _ = server.accept()
            try:
                data = b""
                while b"\n\n" not in data:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                if not data.strip():
                    # handler.py's readiness probe connects and closes without
                    # sending anything — not an error, just nothing to do.
                    conn.close()
                    continue
                request = json.loads(data.decode().strip())
                print(f"Job received: {request.get('job_id')}", flush=True)
                result = self.generate(request)
            except Exception as e:  # noqa: BLE001 — must report back over the socket, not crash the server
                traceback.print_exc()
                result = {"success": False, "error": str(e)}
            finally:
                try:
                    conn.sendall((json.dumps(result) + "\n").encode())
                except Exception:
                    pass
                conn.close()


def main():
    server = ModelServer()
    server.load_model()
    server.run()


if __name__ == "__main__":
    main()
