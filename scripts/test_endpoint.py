#!/usr/bin/env python3
"""
Smoke-test the DreamX-Creator RunPod Serverless endpoint: submits one
image+prompt job via /run, polls /status/{id} until it finishes, and prints
the result (video_url, timings, etc.) or the error.

Usage:
    export RUNPOD_API_KEY=...          # never pass this on the CLI (shell history)
    python3 scripts/test_endpoint.py \\
        --image https://example.com/first_frame.png \\
        --prompt "A cat walking gracefully through the scene, birds chirping"

    # Local image file instead of a URL (base64-encoded automatically):
    python3 scripts/test_endpoint.py --image ./frame.jpg --prompt "..."

Endpoint ID defaults to l4v8b2w427mnlw (this worker's endpoint); override
with --endpoint-id or $RUNPOD_ENDPOINT_ID if you deploy a new one.

See handler.py's generate_video() for the exact input contract this mirrors,
and README.md's "API Usage" section for parameter docs.
"""
import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_ENDPOINT_ID = "l4v8b2w427mnlw"
POLL_INTERVAL_S = 5
# Matches README's suggested Serverless endpoint timeout — see "RunPod Setup".
DEFAULT_MAX_WAIT_S = 1800


def load_image_input(image: str) -> str:
    """Pass URLs through as-is; base64-encode local files (handler.py accepts both)."""
    if image.startswith(("http://", "https://")):
        return image
    path = Path(image)
    if not path.exists():
        print(f"error: image file not found: {image}", file=sys.stderr)
        sys.exit(1)
    return base64.b64encode(path.read_bytes()).decode()


def submit_job(base_url: str, headers: dict, payload: dict) -> str:
    resp = requests.post(f"{base_url}/run", headers=headers, json={"input": payload}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("id")
    if not job_id:
        print(f"error: no job id in /run response: {data}", file=sys.stderr)
        sys.exit(1)
    print(f"Job submitted: {job_id} (status: {data.get('status')})")
    return job_id


def poll_job(base_url: str, headers: dict, job_id: str, max_wait_s: int) -> dict:
    start = time.time()
    last_status = None
    while time.time() - start < max_wait_s:
        resp = requests.get(f"{base_url}/status/{job_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status != last_status:
            print(f"[{time.time()-start:6.1f}s] status: {status}")
            last_status = status
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return data
        time.sleep(POLL_INTERVAL_S)
    print(f"error: still {last_status!r} after {max_wait_s}s, giving up (job keeps running server-side)",
          file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Image URL or path to a local file.")
    parser.add_argument("--prompt", required=True,
                         help="Describes both motion/action AND sound (audio is generated jointly).")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--duration", type=float, default=5.0, help="Clip length in seconds (2.0-8.0).")
    parser.add_argument("--steps", type=int, default=50, help="num_inference_steps (10-100).")
    parser.add_argument("--guidance", type=float, default=5.0, help="Text CFG scale.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--frame-id", default=None)
    parser.add_argument("--endpoint-id", default=os.getenv("RUNPOD_ENDPOINT_ID", DEFAULT_ENDPOINT_ID))
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"),
                         help="Defaults to $RUNPOD_API_KEY — avoid passing secrets as CLI args.")
    parser.add_argument("--max-wait", type=int, default=DEFAULT_MAX_WAIT_S,
                         help=f"Seconds to poll before giving up (default {DEFAULT_MAX_WAIT_S}).")
    parser.add_argument("--sync", action="store_true",
                         help="Use /runsync instead of /run + poll (only works if the job "
                              "finishes within RunPod's ~90s sync response window).")
    args = parser.parse_args()

    if not args.api_key:
        print("error: no API key — pass --api-key or set $RUNPOD_API_KEY", file=sys.stderr)
        sys.exit(1)

    base_url = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}

    payload = {
        "image": load_image_input(args.image),
        "prompt": args.prompt,
        "duration_s": args.duration,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "seed": args.seed,
    }
    if args.negative_prompt:
        payload["negative_prompt"] = args.negative_prompt
    if args.project_id:
        payload["project_id"] = args.project_id
    if args.frame_id:
        payload["frame_id"] = args.frame_id

    start = time.time()
    if args.sync:
        resp = requests.post(f"{base_url}/runsync", headers=headers, json={"input": payload}, timeout=args.max_wait)
        resp.raise_for_status()
        result = resp.json()
    else:
        job_id = submit_job(base_url, headers, payload)
        result = poll_job(base_url, headers, job_id, args.max_wait)
    elapsed = time.time() - start

    print(f"\nFinished in {elapsed:.1f}s. Full response:")
    import json
    print(json.dumps(result, indent=2))

    output = result.get("output", result)
    if isinstance(output, dict) and output.get("error"):
        print(f"\nJob returned an error: {output['error']}", file=sys.stderr)
        sys.exit(1)
    if isinstance(output, dict) and output.get("video_url"):
        print(f"\nvideo_url: {output['video_url']}")


if __name__ == "__main__":
    main()
