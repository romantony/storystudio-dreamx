#!/usr/bin/env python3
"""
Smoke-test the DreamX-Creator 2K refiner endpoint: submits a video_url via /run,
polls until done, prints the result.

    export RUNPOD_API_KEY=...
    export RUNPOD_REFINER_ENDPOINT_ID=...
    python3 scripts/test_refiner.py --video-url https://storyaistudio.app/storystudio/video/..._dreamx_av.mp4
    python3 scripts/test_refiner.py --video-url ... --target-height 1440
"""
import argparse
import json
import os
import sys
import time

from test_endpoint import poll_job, submit_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-url", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--target-height", type=int, help="Output height (skipped if the source is already this tall).")
    group.add_argument("--sr-scale", type=float, help="Upscale factor (1.0-2.25, default 2.0).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--endpoint-id", default=os.getenv("RUNPOD_REFINER_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.getenv("RUNPOD_API_KEY"))
    parser.add_argument("--max-wait", type=int, default=1800)
    args = parser.parse_args()

    if not args.api_key or not args.endpoint_id:
        print("error: set $RUNPOD_API_KEY and $RUNPOD_REFINER_ENDPOINT_ID (or pass the flags)", file=sys.stderr)
        sys.exit(1)

    payload = {"video_url": args.video_url}
    if args.target_height is not None:
        payload["target_height"] = args.target_height
    if args.sr_scale is not None:
        payload["sr_scale"] = args.sr_scale
    if args.seed is not None:
        payload["seed"] = args.seed

    base_url = f"https://api.runpod.ai/v2/{args.endpoint_id}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
    start = time.time()
    result = poll_job(base_url, headers, submit_job(base_url, headers, payload), args.max_wait)
    print(f"\nFinished in {time.time() - start:.1f}s. Full response:")
    print(json.dumps(result, indent=2))

    output = result.get("output", result)
    if isinstance(output, dict) and output.get("error"):
        print(f"\nJob returned an error: {output['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
