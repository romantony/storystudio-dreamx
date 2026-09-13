#!/usr/bin/env python3
"""
Download DreamX-Creator 1.0's base-generator weights onto a RunPod network
volume: the creator/ (7B joint AV DiT), audio_vae/, and wan2.2_ti2v_5b/
(shared video VAE + T5-XXL + tokenizer) directories from GD-ML/DreamX-Creator,
mirrored identically (same paths, same byte sizes, confirmed against both
APIs) on Hugging Face and ModelScope. Total ~43 GB with default flags.

The refiner/ directory (2K refiner, ~11 GB) is NOT fetched by default — this
worker is base-generator-only (see README.md). Pass --include-refiner if you
later add the refiner stage and want its weights alongside these.

Run this on a RunPod CPU/storage pod with the target network volume mounted.

IMPORTANT: interactive/storage Pods often mount the network volume at
/workspace, NOT /runpod-volume (/runpod-volume there is just the pod's local
ephemeral disk and gets wiped when the pod is deleted). The Serverless
endpoint that actually runs this worker mounts the same volume at
/runpod-volume — so as long as you download to the volume's real mount point
on this pod, it will show up at /runpod-volume/dreamx-creator for the
deployed worker. Check `df -h` / `mount` on your pod to confirm where the
network volume is actually mounted before running this.

Two sources are supported via --source:

    modelscope (default — Hugging Face has been very slow for this repo):
        pip install -U modelscope
        python3 scripts/download_weights.py --dest /workspace/dreamx-creator

    hf:
        pip install -U "huggingface_hub[cli]" hf_xet
        python3 scripts/download_weights.py --source hf --dest /workspace/dreamx-creator

Pass --clean to wipe --dest before downloading (e.g. to discard a stalled
partial HF download before switching sources) — it prompts for confirmation
unless --yes is also given.

Safe to re-run otherwise: both backends' snapshot_download skip files that
are already fully downloaded and resume partial ones, so an interrupted run
(or a rerun with --include-refiner added later) won't re-fetch what's
already there.

Speed notes:
- ModelScope: no separate fast-transfer package needed; concurrency is
  controlled by --max-workers (default 8).
- Hugging Face: `creator/video_model`'s two shards and the T5 checkpoint are
  each ~10-11GB single files, and this repo is Xet-enabled. This script
  auto-detects and enables the fastest available transfer backend:
  - `hf_xet` installed -> sets HF_XET_HIGH_PERFORMANCE=1 (current
    huggingface_hub releases use Xet as the fast path; this is the one to
    install today — `pip install hf_xet`).
  - `hf_transfer` installed instead (older huggingface_hub only) ->
    sets HF_HUB_ENABLE_HF_TRANSFER=1.
  - Neither installed -> falls back to a single HTTP connection per file,
    which on many hosts caps out well under the link's real bandwidth.

If downloads are still slow with a fast backend/high concurrency enabled,
the bottleneck is more likely the network volume's own write throughput
(RunPod network volumes are NFS-backed and can be slower than local/
ephemeral disk) rather than the download itself — in that case, download to
local disk first (e.g. --dest /root/dreamx-staging) and copy/rsync to the
network volume path afterward as one large sequential write.
"""
import argparse
import os
import shutil
import time
from pathlib import Path

# GD-ML/DreamX-Creator is Xet-enabled on Hugging Face (confirmed via the HF
# API). Current huggingface_hub releases (>=1.x, and recent 0.3x) use Xet as
# the fast-path transfer backend and have dropped hf_transfer — setting
# HF_HUB_ENABLE_HF_TRANSFER on those versions now only prints a
# FutureWarning and does nothing. Detect which backend is actually
# available and set only the variable that backend honors. (Only relevant
# for --source hf; ModelScope has its own transfer path, tuned via
# --max-workers instead.)
_FAST_TRANSFER = None
try:
    import hf_xet  # noqa: F401
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    _FAST_TRANSFER = "hf_xet (HF_XET_HIGH_PERFORMANCE=1)"
except ImportError:
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        _FAST_TRANSFER = "hf_transfer (HF_HUB_ENABLE_HF_TRANSFER=1)"
    except ImportError:
        pass

REPO_ID = "GD-ML/DreamX-Creator"

# Subdirectories this worker actually loads (see model_server.py's
# build_base_args): creator/ (video+audio DiT + cross-attn weights),
# audio_vae/ (CreatorDACVAE), wan2.2_ti2v_5b/ (video VAE + T5-XXL encoder +
# tokenizer, shared with the upstream Wan2.2-TI2V-5B release).
BASE_PATTERNS = [
    "creator/*",
    "audio_vae/*",
    "wan2.2_ti2v_5b/*",
]
REFINER_PATTERNS = ["refiner/*"]

# Approximate sizes (from the HF repo's blob metadata) for the completion
# report — not load-bearing, just a sanity check against a truncated download.
EXPECTED_MIN_FILES = {
    "creator": 4,          # video_model/{config.json,*.safetensors,*.index.json}, audio_model/*, cross_attn_weights.safetensors
    "audio_vae": 2,
    "wan2.2_ti2v_5b": 5,   # VAE + T5 checkpoint + 3 tokenizer files
}


def dir_stats(path: Path) -> tuple[int, float]:
    if not path.exists():
        return 0, 0.0
    files = [p for p in path.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    return len(files), total_bytes / 1024 ** 3


def clean_dest(dest: Path, yes: bool) -> None:
    if not dest.exists() or not any(dest.iterdir()):
        return
    n_files, size_gb = dir_stats(dest)
    print(f"--clean: {dest} already contains {n_files} files ({size_gb:.2f} GB).")
    if not yes:
        resp = input(f"Delete everything under {dest} before downloading? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted — not deleting, not downloading.")
            raise SystemExit(1)
    print(f"Removing {dest} ...")
    shutil.rmtree(dest)


def download_hf(dest: Path, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    print(f"Source: Hugging Face ({REPO_ID})")
    print(f"Fast transfer: {_FAST_TRANSFER or 'OFF — pip install hf_xet for faster large-file downloads'}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        allow_patterns=patterns,
    )


def download_modelscope(dest: Path, patterns: list[str], max_workers: int) -> None:
    from modelscope import snapshot_download

    print(f"Source: ModelScope ({REPO_ID})")
    print(f"Concurrency: max_workers={max_workers}")
    snapshot_download(
        model_id=REPO_ID,
        local_dir=str(dest),
        allow_patterns=patterns,
        max_workers=max_workers,
    )


def download(dest: Path, include_refiner: bool, source: str, max_workers: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    patterns = list(BASE_PATTERNS)
    if include_refiner:
        patterns += REFINER_PATTERNS

    print(f"Downloading {REPO_ID} -> {dest}")
    for p in patterns:
        print(f"  {p}")

    start = time.time()
    if source == "modelscope":
        download_modelscope(dest, patterns, max_workers)
    else:
        download_hf(dest, patterns)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed/60:.1f} min. Verifying contents of {dest}:\n")
    subdirs = ["creator", "audio_vae", "wan2.2_ti2v_5b"] + (["refiner"] if include_refiner else [])
    total_gb = 0.0
    all_ok = True
    for sub in subdirs:
        n_files, size_gb = dir_stats(dest / sub)
        total_gb += size_gb
        min_expected = EXPECTED_MIN_FILES.get(sub, 1)
        ok = n_files >= min_expected
        all_ok = all_ok and ok
        status = "OK" if ok else "INCOMPLETE"
        print(f"  [{status}] {sub}/: {n_files} files, {size_gb:.2f} GB")

    print(f"\nTotal: {total_gb:.1f} GB")
    if not all_ok:
        print(
            "\nWARNING: one or more directories have fewer files than expected — "
            "the download may have been interrupted. Re-run this script; "
            "snapshot_download resumes rather than starting over."
        )
        raise SystemExit(1)

    print("\nAll expected directories present. MODEL_PATH for the worker should "
          f"point at: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        default=os.getenv("DOWNLOAD_DEST", "/workspace/dreamx-creator"),
        help="Destination directory at the network volume's actual mount point on this pod "
             "(default: $DOWNLOAD_DEST or /workspace/dreamx-creator — NOT /runpod-volume, "
             "which is local pod disk on most interactive Pods). Verify with `df -h` first.",
    )
    parser.add_argument(
        "--include-refiner",
        action="store_true",
        help="Also fetch refiner/ (~11 GB, SR-DiT 5B + upsamplers) for the 2K refiner stage. "
             "Not used by this worker's base-generator-only handler.",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf"],
        default=os.getenv("DOWNLOAD_SOURCE", "modelscope"),
        help="Which host to download from (default: $DOWNLOAD_SOURCE or modelscope — "
             "Hugging Face has been very slow for this repo from some regions). "
             "Both mirror the exact same file layout and byte sizes.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers for --source modelscope (default: 8). Ignored for hf.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete everything already under --dest before downloading (e.g. to discard a "
             "stalled/partial download from a different source). Prompts for confirmation "
             "unless --yes is also passed.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt for --clean.",
    )
    args = parser.parse_args()
    dest = Path(args.dest)
    if args.clean:
        clean_dest(dest, args.yes)
    download(dest, args.include_refiner, args.source, args.max_workers)


if __name__ == "__main__":
    main()
