"""
RunPod Serverless handler for DreamX-Creator's 2K video refiner (SR-DiT, 2x).

Input/output mirror PostProd-Lite's `upscale` mode ({video_url, target_height}
-> {video, upscale}) so quartermaster's upscale step can point here unchanged.
The refiner only synthesizes video; the source clip's audio is copied through.
Model loads once at container start via a persistent model server (same socket
pattern as the base DreamX worker).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

import asyncio
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import boto3
import requests
import runpod

SOCKET_PATH = "/tmp/dreamx_refiner_server.sock"
MODEL_SERVER_SCRIPT = "/workspace/model_server.py"

R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "storyaistudio.app")

# Upstream is trained and benchmarked at 2x. The latent upsampler's scale is
# fixed, so other factors resize the input to target/2 before encoding; keep
# them close to 2x (2.25 still covers 704p -> 1440p).
MIN_SR_SCALE = 1.0
MAX_SR_SCALE = 2.25
DEFAULT_SR_SCALE = 2.0
MAX_INPUT_FRAMES = int(os.getenv("MAX_INPUT_FRAMES", "241"))

model_server_process = None


def start_model_server():
    global model_server_process
    print("Starting refiner model server...")
    model_server_process = subprocess.Popen(
        [sys.executable, "-u", MODEL_SERVER_SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def relay_output():
        while model_server_process.poll() is None:
            line = model_server_process.stdout.readline()
            if line:
                print(f"[Refiner] {line.rstrip()}", flush=True)

    threading.Thread(target=relay_output, daemon=True).start()

    start = time.time()
    while time.time() - start < 900:
        if os.path.exists(SOCKET_PATH):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(SOCKET_PATH)
                sock.close()
                print(f"✓ Refiner server ready (took {time.time() - start:.1f}s)")
                return
            except socket.error:
                pass
        if model_server_process.poll() is not None:
            raise RuntimeError(f"Refiner server crashed with exit code {model_server_process.poll()}")
        time.sleep(1)
    raise RuntimeError("Refiner server failed to start within 900s")


def send_to_model_server(request: dict, timeout: int = 3600) -> dict:
    if model_server_process is not None and model_server_process.poll() is not None:
        raise RuntimeError(f"Refiner server died (exit code {model_server_process.poll()})")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(SOCKET_PATH)
    sock.sendall((json.dumps(request) + "\n\n").encode())

    data = b""
    try:
        while b"\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()

    if not data.strip():
        raise RuntimeError("Empty response from refiner server — it may have crashed")
    return json.loads(data.decode().strip())


def get_r2_client():
    return boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto',
    )


def build_asset_key(asset_type: str, project_id: str = None, frame_id: str = None,
                    job_id: str = None) -> str:
    if project_id and frame_id:
        name = f"{project_id}_{frame_id}_{asset_type}"
    else:
        name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{job_id or 'unknown'}_{asset_type}"
    return f"storystudio/video/{name}.mp4"


def download_video(video_url: str, dest: Path) -> None:
    if f"{R2_PUBLIC_URL}/" in video_url:
        r2_key = video_url.split(f"{R2_PUBLIC_URL}/", 1)[1]
        try:
            get_r2_client().download_file(R2_BUCKET_NAME, r2_key, str(dest))
            return
        except Exception as e:
            print(f"R2 direct fetch failed ({e}), falling back to HTTP")
    with requests.get(video_url, stream=True, timeout=(10, 120)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def probe_video(path: Path) -> Dict[str, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,nb_read_packets", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "frames": int(stream["nb_read_packets"])}


def refine_video(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    start_time = time.time()

    try:
        video_url = job_input.get("video_url")
        if not video_url:
            return {"error": "Missing required input: video_url"}

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = Path(tmpdir) / "input.mp4"
            download_video(video_url, src_path)
            info = probe_video(src_path)
            print(f"Input: {info['width']}x{info['height']}, {info['frames']} frames")

            if info["frames"] > MAX_INPUT_FRAMES:
                return {"error": f"input has {info['frames']} frames; max is {MAX_INPUT_FRAMES}"}

            target_height = job_input.get("target_height")
            if target_height is not None:
                target_height = int(target_height)
                if info["height"] >= target_height:
                    return {"video": video_url, "video_url": video_url, "upscale": "skipped_source_hires"}
                sr_scale = target_height / info["height"]
            else:
                sr_scale = float(job_input.get("sr_scale") or DEFAULT_SR_SCALE)
            if not (MIN_SR_SCALE <= sr_scale <= MAX_SR_SCALE):
                return {"error": f"scale {sr_scale:.3f}x (from {info['height']}p) must be between "
                                 f"{MIN_SR_SCALE} and {MAX_SR_SCALE}"}

            seed = int(job_input.get("seed", 42))
            result = send_to_model_server({
                "job_id": job_id,
                "input_path": str(src_path),
                "sr_scale": sr_scale,
                "seed": seed,
            })
            if not result.get("success"):
                return {"error": result.get("error", "Refinement failed")}

            out_path = Path(result["output_path"])
            out_info = probe_video(out_path)
            r2_key = build_asset_key(f"dreamx_sr_{out_info['height']}p",
                                     job_input.get("project_id"), job_input.get("frame_id"), job_id)
            with open(out_path, "rb") as f:
                get_r2_client().upload_fileobj(f, R2_BUCKET_NAME, r2_key)
            url = f"https://{R2_PUBLIC_URL}/{r2_key}"
            size_mb = round(out_path.stat().st_size / 1024 / 1024, 2)
            shutil.rmtree(out_path.parent, ignore_errors=True)

            return {
                "video": url,
                "video_url": url,
                "upscale": f"dreamx_sr_{out_info['height']}p",
                "width": out_info["width"],
                "height": out_info["height"],
                "sr_scale": round(sr_scale, 4),
                "seed": seed,
                "video_size_mb": size_mb,
                "model_generation_time": round(result.get("generation_time", 0), 2),
                "generation_time": round(time.time() - start_time, 2),
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def async_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    # Refinement takes minutes; keep it off the RunPod SDK's event loop.
    return await asyncio.to_thread(refine_video, job)


print("=" * 60)
print("DreamX-Creator 2K Refiner (SR-DiT) Handler")
print("=" * 60)
start_model_server()
print("✓ Handler ready with warm refiner")

runpod.serverless.start({"handler": async_handler})
