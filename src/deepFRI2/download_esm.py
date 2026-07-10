#!/usr/bin/env python
"""Download the ESM weights used by deepFRI2 into a local directory.

Current deepFRI2 release uses ``facebook/esm2_t33_650M_UR50D`` (~2.5 GB) to turn sequences into
per-residue embeddings. Rather than relying on the HuggingFace cache at run time,
this script fetches the weights once into a repo-local folder so inference can run
fully offline afterwards (the notebook / inference scripts load with
``local_files_only=True``).

Usage
-----
    python src/deepFRI2/download_esm.py                  # default model into <repo>/params/<model>
    python src/deepFRI2/download_esm.py --dest /some/dir # custom destination root
    python src/deepFRI2/download_esm.py --model facebook/esm2_t30_150M_UR50D
    python src/deepFRI2/download_esm.py --force          # re-download even if present

The download is idempotent: an already-complete folder is left untouched unless
``--force`` is given. Only the files needed for inference (config, tokenizer,
safetensors weights) are fetched.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Files required to load the model + tokenizer for inference.
ALLOW_PATTERNS = ["*.json", "*.txt", "*.safetensors"]
DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
# Files that must exist for the folder to count as a complete download.
REQUIRED_FILES = ["config.json", "model.safetensors"]


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _is_complete(dest: Path) -> bool:
    return dest.is_dir() and all((dest / name).exists() for name in REQUIRED_FILES)


def download_esm(model: str = DEFAULT_MODEL,
                 dest_root: Path = None,
                 force: bool = False) -> Path:
    """Download ``model`` into ``dest_root/<model basename>`` and return that path."""
    if dest_root is None:
        # download_esm.py lives at <repo>/src/deepFRI2/; params/ is at the repo root.
        dest_root = Path(__file__).resolve().parents[2] / "params"
    dest = dest_root / model.split("/")[-1]

    if _is_complete(dest) and not force:
        size_gb = _dir_size_bytes(dest) / 1e9
        print(f"ESM weights already present at {dest} ({size_gb:.2f} GB). "
              f"Use --force to re-download.")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model} -> {dest} ...")
    snapshot_download(
        repo_id=model,
        local_dir=str(dest),
        allow_patterns=ALLOW_PATTERNS,
        force_download=force,
    )

    if not _is_complete(dest):
        missing = [name for name in REQUIRED_FILES if not (dest / name).exists()]
        raise RuntimeError(f"Download incomplete; missing: {missing}")

    size_gb = _dir_size_bytes(dest) / 1e9
    print(f"Done. {model} is now available locally at {dest} ({size_gb:.2f} GB).")
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace repo id (default: {DEFAULT_MODEL}).")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Destination root directory (default: <repo>/params).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the weights are already present.")
    args = parser.parse_args(argv)

    download_esm(model=args.model, dest_root=args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
