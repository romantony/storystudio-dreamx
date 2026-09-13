"""
RunPod Serverless Handler for DreamX-Creator 1.0 — base audio-video generator
(image + prompt -> synced 5s video+audio). No 2K refiner in this worker.

Model loads ONCE on container startup via a persistent model server and
stays warm between jobs. Structure mirrors
wan22-14B-fp8-4steps/handler_v2.py (socket IPC to model_server.py, R2
upload via storystudio/{category}/... naming), adapted for DreamX-Creator's
image+prompt+duration input and joint video+audio output.
"""
# FIRST: Set CUDA environment variables before ANY imports
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

import sys
import random
import asyncio
import runpod
import subprocess
import tempfile
import base64
import time
import socket
import json
import boto3
import requests
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/runpod-volume/dreamx-creator")
SOCKET_PATH = "/tmp/dreamx_model_server.sock"
MODEL_SERVER_SCRIPT = "/workspace/model_server.py"

# R2 Configuration — no hardcoded fallbacks for credentials/account: these
# must come from the endpoint's env, same as the Wan2.2 worker.
R2_ACCOUNT_ID = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "storyaistudio.app")

# DreamX-Creator's Verse-Bench default/only officially-tested duration is 5s;
# the pipeline accepts any float duration but longer/shorter clips are
# unverified against the model's training distribution. Clamp to a sane
# range rather than pass through untested values.
MIN_DURATION_S = 2.0
MAX_DURATION_S = 8.0

# Model is NOT distilled — 50 steps is the published default. Fewer steps is
# allowed (no hard block, same posture as the Wan2.2 worker's sample_steps
# validation) but is unverified for quality since no distilled DreamX
# checkpoint has been released yet.
MIN_STEPS = 10
MAX_STEPS = 100
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "50"))

# Global state
model_server_process = None


def _read(path: str):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _gb(value) -> str:
    try:
        return f"{int(value) / 1024 ** 3:.1f}GB"
    except (TypeError, ValueError):
        return str(value)


def resource_snapshot() -> str:
    parts = []
    usage, limit = _read("/sys/fs/cgroup/memory.current"), _read("/sys/fs/cgroup/memory.max")
    stat_path, anon_key, file_key = "/sys/fs/cgroup/memory.stat", "anon", "file"
    if usage is None:  # cgroup v1
        usage = _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        limit = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        stat_path, anon_key, file_key = "/sys/fs/cgroup/memory/memory.stat", "total_rss", "total_cache"
    if usage is not None:
        parts.append(f"cgroup_mem={_gb(usage)}/{_gb(limit)}")
    stat = dict(line.split() for line in (_read(stat_path) or "").splitlines() if len(line.split()) == 2)
    if stat:
        parts.append(f"anon={_gb(stat.get(anon_key))} file_cache={_gb(stat.get(file_key))}")
    meminfo = dict(line.split(":", 1) for line in (_read("/proc/meminfo") or "").splitlines() if ":" in line)
    if meminfo:
        parts.append(f"host_avail={meminfo['MemAvailable'].strip()}/{meminfo['MemTotal'].strip()}")
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        parts.append(f"gpu={gpu}")
    except (OSError, subprocess.SubprocessError):
        pass
    return " | ".join(parts)


def start_resource_monitor(interval_s: int = 5):
    # Workers have been receiving SIGTERM with no error reported; log memory
    # so the last lines before "Kill worker." show whether it's memory pressure.
    import threading

    def loop():
        while True:
            print(f"[Resources] {resource_snapshot()}", flush=True)
            time.sleep(interval_s)

    threading.Thread(target=loop, daemon=True).start()


def verify_model_present():
    """Confirm the Creator generator + T5/VAE deps exist on the network volume."""
    required = Path(MODEL_PATH) / "wan2.2_ti2v_5b" / "models_t5_umt5-xxl-enc-bf16.pth"
    if not required.exists():
        raise RuntimeError(
            f"Model weights not found at {MODEL_PATH}. "
            "Download GD-ML/DreamX-Creator's creator/, audio_vae/, and "
            "wan2.2_ti2v_5b/ directories to the network volume before starting "
            "(see README.md)."
        )
    print(f"✓ Model found at {MODEL_PATH}")


def start_model_server():
    """Start the persistent model server as a background process."""
    global model_server_process

    if model_server_process is not None:
        print("Model server already running")
        return

    print("Starting persistent model server...")

    model_server_process = subprocess.Popen(
        [sys.executable, "-u", MODEL_SERVER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    import threading

    def read_model_server_output():
        while model_server_process is not None and model_server_process.poll() is None:
            if model_server_process.stdout:
                line = model_server_process.stdout.readline()
                if line:
                    print(f"[ModelServer] {line.rstrip()}", flush=True)

    output_thread = threading.Thread(target=read_model_server_output, daemon=True)
    output_thread.start()

    print("Waiting for model to load into GPU memory...")
    start_time = time.time()
    max_wait = 900  # model load involves a 55GB checkpoint set from network volume

    while time.time() - start_time < max_wait:
        if os.path.exists(SOCKET_PATH):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(SOCKET_PATH)
                sock.close()
                print(f"✓ Model server ready! (took {time.time() - start_time:.1f}s)")
                return
            except socket.error:
                pass

        if model_server_process.poll() is not None:
            raise Exception(f"Model server crashed with exit code: {model_server_process.poll()}")

        time.sleep(1)

    raise Exception("Model server failed to start within timeout")


def send_to_model_server(request: dict, timeout: int = 3600) -> dict:
    """Send a job request to the model server with robust error handling."""
    global model_server_process

    if model_server_process is not None:
        poll = model_server_process.poll()
        if poll is not None:
            print(f"WARNING: Model server process died with code {poll}")
            if model_server_process.stdout:
                remaining = model_server_process.stdout.read()
                if remaining:
                    print(f"Model server final output: {remaining}")
            raise Exception(f"Model server process died unexpectedly (exit code: {poll})")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(SOCKET_PATH)
    except socket.error as e:
        raise Exception(f"Failed to connect to model server: {e}. Server may have crashed.")

    message = json.dumps(request) + "\n\n"
    sock.sendall(message.encode())
    print(f"Request sent to model server ({len(message)} bytes)", flush=True)

    data = b""
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
    except socket.timeout:
        sock.close()
        raise Exception(f"Socket read timeout after {timeout}s - generation may still be running")

    sock.close()

    response_str = data.decode().strip()
    if not response_str:
        if model_server_process is not None:
            poll = model_server_process.poll()
            if poll is not None:
                raise Exception(f"Model server crashed during generation (exit code: {poll})")
        raise Exception("Empty response from model server - server may have crashed or timed out")

    try:
        return json.loads(response_str)
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON from model server: {e}. Response was: {response_str[:500]}")


def get_r2_client():
    return boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto',
    )


def build_asset_key(category: str, asset_type: str, ext: str,
                     project_id: str = None, frame_id: str = None,
                     job_id: str = None) -> str:
    """storystudio/{category}/{project_id}_{frame_id}_{asset_type}.{ext}, or
    storystudio/{category}/{timestamp}_{job_id}_{asset_type}.{ext} when
    project_id/frame_id (both optional request fields) aren't given."""
    if project_id and frame_id:
        name = f"{project_id}_{frame_id}_{asset_type}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        name = f"{timestamp}_{job_id or 'unknown'}_{asset_type}"
    return f"storystudio/{category}/{name}.{ext}"


def upload_to_r2(file_path: str, r2_key: str) -> str:
    s3_client = get_r2_client()
    with open(file_path, 'rb') as f:
        s3_client.upload_fileobj(f, R2_BUCKET_NAME, r2_key)
    return f"https://{R2_PUBLIC_URL}/{r2_key}"


def generate_video(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler for DreamX-Creator joint audio-video generation."""
    job_input = job.get("input", {})
    job_id = job.get("id", "unknown")
    start_time = time.time()

    try:
        if "image" not in job_input:
            return {"error": "Missing required input: image"}
        if "prompt" not in job_input or not job_input["prompt"]:
            return {"error": "Missing required input: prompt"}

        image_input = job_input["image"]
        prompt = job_input["prompt"]
        negative_prompt = job_input.get("negative_prompt")
        duration = float(job_input.get("duration_s", 5.0))
        num_inference_steps = int(job_input.get("num_inference_steps") or DEFAULT_STEPS)
        guidance_scale = float(job_input.get("guidance_scale", 5.0))
        # A fixed default seed would make every request with the same image and
        # prompt produce the same clip; pick one per job and return it instead.
        seed = job_input.get("seed")
        seed = random.randint(0, 2**31 - 1) if seed is None else int(seed)
        print(f"Seed: {seed} | steps: {num_inference_steps}")

        if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
            return {"error": f"duration_s must be between {MIN_DURATION_S} and {MAX_DURATION_S}"}
        if not (MIN_STEPS <= num_inference_steps <= MAX_STEPS):
            return {"error": f"num_inference_steps must be between {MIN_STEPS} and {MAX_STEPS}"}

        # Handle image input (URL or base64) — same R2-direct-fetch shortcut
        # as the Wan2.2 worker, to skip the app-server hop for our own assets.
        if image_input.startswith(('http://', 'https://')):
            t0 = time.time()
            r2_fetched = False
            if f"{R2_PUBLIC_URL}/" in image_input:
                r2_key = image_input.split(f"{R2_PUBLIC_URL}/", 1)[1]
                try:
                    print(f"Downloading image from R2 directly: {r2_key}")
                    obj = get_r2_client().get_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
                    image_bytes = obj["Body"].read()
                    r2_fetched = True
                except Exception as r2_err:
                    print(f"R2 direct fetch failed ({r2_err}), falling back to HTTP")
            if not r2_fetched:
                print(f"Downloading image from URL: {image_input}")
                response = requests.get(image_input, timeout=(10, 60))
                response.raise_for_status()
                image_bytes = response.content
            print(f"Image ready: {len(image_bytes)/1024:.0f} KB in {time.time()-t0:.1f}s")
        else:
            image_bytes = base64.b64decode(image_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "input_image.jpg"
            image_path.write_bytes(image_bytes)

            output_path = Path(tmpdir) / "output.mp4"

            print("Sending job to warm model server...")
            request = {
                "job_id": job_id,
                "image_path": str(image_path),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "duration": duration,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "output_path": str(output_path),
            }

            result = send_to_model_server(request)

            if not result.get("success"):
                return {"error": result.get("error", "Generation failed")}

            if not output_path.exists():
                return {"error": "Model server reported success but output file is missing"}

            video_size_mb = round(output_path.stat().st_size / 1024 / 1024, 2)

            print("Uploading video to R2...")
            r2_key = build_asset_key(
                "video", "dreamx_av", "mp4",
                job_input.get("project_id"), job_input.get("frame_id"), job_id,
            )
            video_url = upload_to_r2(str(output_path), r2_key)

            generation_time = time.time() - start_time

            return {
                "video_url": video_url,
                "generation_time": round(generation_time, 2),
                "model_generation_time": round(result.get("generation_time", 0), 2),
                "video_size_mb": video_size_mb,
                "duration_s": duration,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
            }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def async_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    # The RunPod SDK runs job-take and its HTTP session on one asyncio loop and
    # calls sync handlers directly on it, so a multi-minute blocking handler
    # stalls that loop for the whole generation. Run the work in a thread.
    return await asyncio.to_thread(generate_video, job)


print("=" * 60)
print("DreamX-Creator 1.0 Base Audio-Video Handler — RTX 6000 Ada")
print("=" * 60)

start_resource_monitor()
verify_model_present()
start_model_server()

print("✓ Handler ready with warm model!")
print("=" * 60)

runpod.serverless.start({"handler": async_handler})
