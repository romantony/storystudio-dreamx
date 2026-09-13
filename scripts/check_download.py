#!/usr/bin/env python3
"""
Check download progress of DreamX-Creator weights against the exact file
manifest from GD-ML/DreamX-Creator — the paths and byte sizes below are
identical on Hugging Face and ModelScope (confirmed against both APIs), so
this check works regardless of which --source download_weights.py used.
Run this anytime, including while download_weights.py is still running, to
see what's done, what's partial, and what's still missing.

Usage (on the pod, pointed at the same --dest used for download_weights.py):
    python3 scripts/check_download.py --dest /workspace/dreamx-creator
    python3 scripts/check_download.py --dest /workspace/dreamx-creator --include-refiner

Exit code is 0 if everything in scope is fully downloaded, 1 otherwise, so
this can be used as a readiness gate in a script (e.g. `until python3
scripts/check_download.py --dest ...; do sleep 30; done`).
"""
import argparse
import os
from pathlib import Path

# Exact byte sizes from the repo's blob metadata (HF API with blobs=true),
# fetched 2026-09-12. Re-fetch and update this table if the upstream repo
# ever republishes these files with different content.
MANIFEST = {
    "audio_vae/config.json": 471,
    "audio_vae/diffusion_pytorch_model.safetensors": 743102794,
    "creator/audio_model/config.json": 691,
    "creator/audio_model/diffusion_pytorch_model.safetensors": 5676857112,
    "creator/cross_attn_weights.safetensors": 2556452768,
    "creator/merged_lora_info.json": 427,
    "creator/video_model/config.json": 786,
    "creator/video_model/diffusion_pytorch_model-00001-of-00002.safetensors": 10658247312,
    "creator/video_model/diffusion_pytorch_model-00002-of-00002.safetensors": 9340987096,
    "creator/video_model/diffusion_pytorch_model.safetensors.index.json": 72864,
    "wan2.2_ti2v_5b/google/umt5-xxl/special_tokens_map.json": 6623,
    "wan2.2_ti2v_5b/google/umt5-xxl/spiece.model": 4548313,
    "wan2.2_ti2v_5b/google/umt5-xxl/tokenizer_config.json": 61728,
    "wan2.2_ti2v_5b/google/umt5-xxl/tokenizer.json": 16837417,
    "wan2.2_ti2v_5b/models_t5_umt5-xxl-enc-bf16.pth": 11361920418,
    "wan2.2_ti2v_5b/Wan2.2_VAE.pth": 2818839170,
}
REFINER_MANIFEST = {
    "refiner/latent_upsampler_2d_causal.pt": 463683163,
    "refiner/latent_upsampler_flash.pt": 20014099,
    "refiner/lightvae_nu_scheme3.pt": 590511261,
    "refiner/sr_dit_5b.pt": 9999839408,
}


def human(n: int) -> str:
    return f"{n / 1024**3:.2f} GB" if n >= 1024**3 else f"{n / 1024**2:.1f} MB"


def check(dest: Path, include_refiner: bool) -> bool:
    manifest = dict(MANIFEST)
    if include_refiner:
        manifest.update(REFINER_MANIFEST)

    total_expected = sum(manifest.values())
    total_present = 0
    all_complete = True
    missing, partial = [], []

    print(f"Checking {dest} against {len(manifest)} expected files "
          f"({human(total_expected)} total)\n")

    for rel_path, expected_size in sorted(manifest.items()):
        p = dest / rel_path
        if not p.exists():
            missing.append((rel_path, expected_size))
            all_complete = False
            continue
        actual_size = p.stat().st_size
        total_present += min(actual_size, expected_size)
        if actual_size < expected_size:
            partial.append((rel_path, actual_size, expected_size))
            all_complete = False
        elif actual_size > expected_size:
            # huggingface_hub never overshoots a completed file; a mismatch
            # this direction means the upstream file changed since MANIFEST
            # was captured, not a download problem.
            print(f"  [SIZE MISMATCH] {rel_path}: {human(actual_size)} on disk, "
                  f"expected {human(expected_size)} — manifest may be stale")

    pct = 100.0 * total_present / total_expected if total_expected else 100.0
    print(f"Progress: {human(total_present)} / {human(total_expected)} ({pct:.1f}%)\n")

    if partial:
        print(f"IN PROGRESS / INCOMPLETE ({len(partial)}):")
        for rel_path, actual, expected in partial:
            file_pct = 100.0 * actual / expected if expected else 0.0
            print(f"  {rel_path}: {human(actual)} / {human(expected)} ({file_pct:.1f}%)")
        print()

    if missing:
        print(f"NOT STARTED ({len(missing)}):")
        for rel_path, expected in missing:
            print(f"  {rel_path}: {human(expected)}")
        print()

    if all_complete:
        print("All expected files fully downloaded.")
    else:
        remaining = total_expected - total_present
        print(f"Remaining: {human(remaining)} across {len(partial) + len(missing)} file(s).")

    return all_complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        default=os.getenv("DOWNLOAD_DEST", "/workspace/dreamx-creator"),
        help="Directory to check (same --dest used for download_weights.py).",
    )
    parser.add_argument(
        "--include-refiner",
        action="store_true",
        help="Also check refiner/ files (only relevant if you downloaded them with --include-refiner).",
    )
    args = parser.parse_args()
    ok = check(Path(args.dest), args.include_refiner)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
