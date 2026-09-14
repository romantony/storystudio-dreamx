"""Persistent model server for DreamX-Creator's 2K video refiner (SR-DiT).

Upstream video_refiner/inference_sr.py is a one-shot CLI whose model setup runs
at module level. Rather than re-implementing it, its source is executed
unmodified in two parts: everything before the inference loop runs once at
startup (checkpoint load, window attention, LQ anchor, prompt-embedding cache),
and the loop runs per job against a one-video `samples` list. Only
`write_video` is swapped, for a CRF encoder: torchvision's default bitrate is
far too low for 2K output.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

REFINER_ROOT = "/workspace/dreamx-creator/video_refiner"
SOCKET_PATH = "/tmp/dreamx_refiner_server.sock"
LOOP_MARKER = "# ── Inference loop ──"
# Setup's "gather input videos" step only records this path; it is never read.
PLACEHOLDER_INPUT = "/tmp/refiner_placeholder.mp4"


def int_env(name: str, default):
    """Positive int from the environment; blank or placeholder text means unset."""
    raw = os.getenv(name, "").strip()
    if raw.lower() in ("", "unset", "none", "default"):
        return default
    if not raw.isdigit() or int(raw) < 1:
        raise RuntimeError(f"{name} must be a positive integer or left unset, got {raw!r}")
    return int(raw)


REFINER_FAST = os.getenv("REFINER_FAST", "0").strip() == "1"
WINDOW_CHUNK = int_env("REFINER_WINDOW_CHUNK", None)
# Rolling KV cache length in 3-latent-frame chunks. VRAM grows linearly with it:
# upstream's 9 OOMs a 48GB card at 2496x1408 after 2 chunks (upstream suggests 6
# for 960x1664, on a 96GB H20). Fewer chunks = less temporal context per chunk.
KV_LEN = int_env("REFINER_KV_LEN", 3)


def build_cli_args() -> list[str]:
    """Mirrors video_refiner/run_inference.sh defaults (paths resolve through the
    checkpoints -> network volume symlink created in the Dockerfile)."""
    args = [
        "--config_path", "configs/sr_dit_5b.yaml",
        "--checkpoint_path", "../checkpoints/refiner/sr_dit_5b.pt",
        "--input_path", PLACEHOLDER_INPUT,
        "--output_folder", "/tmp/refiner_setup",
        "--sigma_start", "0.6251",
        "--num_frames", "-1",
        "--auto_target_size",
        "--sr_scale", "2.0",
        "--causal",
        "--seed", "42",
        "--kv_len", str(KV_LEN),
        "--latent_upsampler_config", "configs/latent_upsampler_flash.yaml",
        "--latent_upsampler_ckpt", "../checkpoints/refiner/latent_upsampler_flash.pt",
        "--use_window_attn",
        "--window_attn_impl", "triton",
        "--window_block_hw", "4", "4",
        "--window_block_radius_hw", "3", "3",
        "--use_lq_anchor",
        "--lq_guidance_mode", "v",
        "--lq_guidance_scale", "1.0",
        "--lq_anchor_align", "frame",
    ]
    if WINDOW_CHUNK:
        args += ["--window_chunk", str(WINDOW_CHUNK)]
    if REFINER_FAST:
        # Upstream benchmark: 3.4x faster end to end, ~36 dB PSNR vs the bf16 default.
        args += ["--enable_nu_lightvae", "--nu_lightvae_type", "scheme3",
                 "--fp8_linear", "--fp8_targets", "all"]
    return args


def write_video_crf(output_path, video, fps):
    """Drop-in for torchvision.io.write_video(path, uint8 [T,H,W,C], fps)."""
    t, h, w, _ = video.shape
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(float(fps)), "-i", "-",
         "-c:v", "libx264", "-crf", os.getenv("VIDEO_CRF", "18"), "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", output_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Stream frame by frame: a full 2K clip is >1GB of raw RGB.
    for i in range(t):
        proc.stdin.write(video[i].contiguous().numpy().tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg encode failed: {stderr.strip()}")


class RefinerServer:
    def __init__(self):
        self.ns = None
        self.loop_code = None

    def load(self):
        os.chdir(REFINER_ROOT)
        sys.path.insert(0, REFINER_ROOT)

        volume = Path("/runpod-volume")
        if not volume.is_dir() or not any(volume.iterdir()):
            raise RuntimeError(
                "/runpod-volume is missing or empty — attach the DreamX network volume "
                "to this endpoint (Edit Endpoint → Advanced → Network Volume).")
        for rel in ("refiner/sr_dit_5b.pt", "refiner/latent_upsampler_flash.pt",
                    "wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth", "wan2.2_ti2v_5b/Wan2.2_VAE.pth"):
            if not (Path("../checkpoints") / rel).exists():
                listing = sorted(p.name for p in volume.iterdir())
                raise RuntimeError(
                    f"Missing {rel} under /runpod-volume/dreamx-creator (volume root has: {listing}) — "
                    "download refiner/ with scripts/download_weights.py --include-refiner (see README.md).")
        if REFINER_FAST and not Path("../checkpoints/refiner/lightvae_nu_scheme3.pt").exists():
            raise RuntimeError("REFINER_FAST=1 needs refiner/lightvae_nu_scheme3.pt on the volume")

        src = Path("inference_sr.py").read_text()
        idx = src.find(LOOP_MARKER)
        if idx < 0:
            raise RuntimeError(f"'{LOOP_MARKER}' not found in inference_sr.py — upstream changed")
        setup_src, loop_src = src[:idx], src[idx:]
        # Pad so tracebacks from the loop report real inference_sr.py line numbers.
        self.loop_code = compile("\n" * setup_src.count("\n") + loop_src, "inference_sr.py", "exec")

        print(f"Refiner mode: {'fast (fp8 + LightVAE-NU)' if REFINER_FAST else 'default (bf16 + full Wan2.2 VAE)'}")
        t0 = time.time()
        sys.argv = ["inference_sr.py", *build_cli_args()]
        self.ns = {"__name__": "inference_sr"}
        exec(compile(setup_src, "inference_sr.py", "exec"), self.ns)
        self.ns["write_video"] = write_video_crf

        torch = self.ns["torch"]
        decode_timed = self.ns["_decode_timed"]

        def decode_with_dit_offloaded(pipeline, latent):
            # 2K VAE decode OOMs a 48GB card while the DiT (~10 GB) and the last
            # chunk's KV cache still sit on the GPU; neither is needed to decode.
            t_off = time.time()
            pipeline.kv_caches = None
            pipeline.generator.to("cpu")
            torch.cuda.empty_cache()
            print(f"Offloaded DiT for decode in {time.time() - t_off:.1f}s", flush=True)
            try:
                return decode_timed(pipeline, latent)
            finally:
                t_on = time.time()
                pipeline.generator.to("cuda")
                print(f"Restored DiT in {time.time() - t_on:.1f}s", flush=True)

        # The loop resolves _decode_timed from this namespace at call time.
        self.ns["_decode_timed"] = decode_with_dit_offloaded
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"✓ Refiner ready in {time.time() - t0:.1f}s — {allocated:.1f} GB allocated", flush=True)

    def refine(self, params: dict) -> dict:
        t0 = time.time()
        ns = self.ns
        args = ns["args"]
        out_dir = tempfile.mkdtemp(prefix="refiner_out_")
        args.output_folder = out_dir
        args.sr_scale = float(params["sr_scale"])
        ns["samples"] = [(params["input_path"], args.prompt)]
        torch = ns["torch"]
        torch.manual_seed(int(params["seed"]))
        torch.cuda.reset_peak_memory_stats()

        try:
            exec(self.loop_code, ns)
            outputs = sorted(Path(out_dir).glob("*_sr.mp4"))
            if len(outputs) != 1:
                raise RuntimeError(f"expected one refined output in {out_dir}, found {len(outputs)}")
        except BaseException:
            shutil.rmtree(out_dir, ignore_errors=True)
            raise
        finally:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"Peak VRAM {peak:.1f} / {total:.1f} GB (kv_len={KV_LEN}, sr_scale={args.sr_scale})", flush=True)
            # The pipeline keeps the last clip's KV cache as an attribute; drop it
            # so an idle worker (or one that just OOM'd) holds only the weights.
            if getattr(ns.get("pipeline"), "kv_caches", None) is not None:
                ns["pipeline"].kv_caches = None
            torch.cuda.empty_cache()
        return {
            "success": True,
            "output_path": str(outputs[0]),
            "generation_time": time.time() - t0,
        }

    def run(self):
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_PATH)
        server.listen(1)
        print(f"Refiner server listening on {SOCKET_PATH}", flush=True)

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
                    # handler.py's readiness probe connects and closes without sending anything.
                    conn.close()
                    continue
                request = json.loads(data.decode().strip())
                print(f"Job received: {request.get('job_id')}", flush=True)
                result = self.refine(request)
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
    server = RefinerServer()
    server.load()
    server.run()


if __name__ == "__main__":
    main()
