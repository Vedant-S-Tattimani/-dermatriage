"""
HAM10000 dataset download script — uses kagglehub API.

Usage:
    pip install "kagglehub[pandas-datasets]"
    python ml/download_dataset.py

The dataset files are cached by kagglehub at:
    ~/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/

A symlink / copy is also placed in data/raw/ for convenience.
"""
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SLUG = "kmader/skin-cancer-mnist-ham10000"


def download():
    try:
        import kagglehub
    except ImportError:
        print("ERROR: 'kagglehub' not installed.")
        print("  Run: pip install \"kagglehub[pandas-datasets]\"")
        sys.exit(1)

    print(f"Downloading HAM10000 via kagglehub …")
    print(f"  Dataset : {DATASET_SLUG}")

    # Download and get local cache path
    dataset_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"  Cached at: {dataset_path}")

    # ── Copy / link key assets into data/raw/ ──────────────────────────────
    copied = 0
    for item in dataset_path.rglob("*"):
        if item.is_file():
            dest = RAW_DIR / item.name
            if not dest.exists():
                shutil.copy2(item, dest)
                copied += 1

    print(f"  Copied {copied} files to {RAW_DIR}")

    # ── Merge image parts into a single images/ folder ─────────────────────
    images_dir = RAW_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    moved = 0
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        part_dir = RAW_DIR / part
        if part_dir.exists():
            for img in part_dir.glob("*.jpg"):
                dest = images_dir / img.name
                if not dest.exists():
                    img.rename(dest)
                    moved += 1
            print(f"  Merged {moved} images from {part}")

    print(f"\n✅ Download complete!")
    print(f"   Images : {images_dir}")
    print(f"   Metadata CSV: {RAW_DIR / 'HAM10000_metadata.csv'}")
    return dataset_path


def load_metadata_df(file_path: str = "HAM10000_metadata.csv"):
    """
    Load any CSV from the dataset directly into a pandas DataFrame
    using the kagglehub pandas adapter (no manual file management needed).

    Args:
        file_path: Relative path to the file within the dataset archive.
                   Use "" to list available files.
    """
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError:
        print("ERROR: 'kagglehub' not installed.")
        print("  Run: pip install \"kagglehub[pandas-datasets]\"")
        sys.exit(1)

    print(f"Loading '{file_path}' from {DATASET_SLUG} via kagglehub pandas adapter …")
    # Use dataset_load() — load_dataset() is deprecated as of kagglehub 0.3+
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        DATASET_SLUG,
        file_path,
    )
    print(f"  Loaded {len(df):,} rows | columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download HAM10000 via kagglehub")
    parser.add_argument(
        "--load-csv",
        metavar="FILE",
        default=None,
        help="Also load a specific CSV into pandas (e.g. HAM10000_metadata.csv)",
    )
    args = parser.parse_args()

    download()

    if args.load_csv:
        df = load_metadata_df(args.load_csv)
        print("\nFirst 5 records:")
        print(df.head())
