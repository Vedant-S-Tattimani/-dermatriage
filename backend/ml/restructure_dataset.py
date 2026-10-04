"""
Restructures the HAM10000 dataset into a high-quality train/val/test structure.
- Prevents data leakage by splitting by lesion_id.
- Maintains class balance via stratified splitting.
- Generates a detailed statistics report.
"""
import os
import json
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
METADATA_CSV = RAW_DIR / "metadata.csv"
STATS_PATH = PROCESSED_DIR / "dataset_summary.json"

CLASS_NAMES = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma"
}

def find_image(image_id: str, raw_dir: Path) -> Path:
    """Finds an image file in the raw directory."""
    search_dirs = [raw_dir / "HAM10000_images", raw_dir]
    for d in search_dirs:
        if not d.exists(): continue
        for ext in [".jpg", ".jpeg", ".png"]:
            p = d / f"{image_id}{ext}"
            if p.exists(): return p
    return None

def main():
    if not METADATA_CSV.exists():
        print(f"ERROR: Metadata not found at {METADATA_CSV}")
        return

    print("Loading and analyzing metadata...")
    df = pd.read_csv(METADATA_CSV)
    
    # ── 1. PREVENT DATA LEAKAGE ──────────────────────────────────────────────
    # We must split by lesion_id, not image_id.
    # First, get a dataframe of unique lesions with their diagnosis.
    lesions_df = df.drop_duplicates(subset="lesion_id").copy()
    print(f"Found {len(df)} total images for {len(lesions_df)} unique lesions.")

    # ── 2. STRATIFIED SPLIT (75/15/10) ───────────────────────────────────────
    train_lesions, val_test_lesions = train_test_split(
        lesions_df, test_size=0.25, stratify=lesions_df["dx"], random_state=42
    )
    val_lesions, test_lesions = train_test_split(
        val_test_lesions, test_size=0.40, stratify=val_test_lesions["dx"], random_state=42
    )

    # Map lesion IDs to split names
    split_map = {}
    for lid in train_lesions["lesion_id"]: split_map[lid] = "train"
    for lid in val_lesions["lesion_id"]:   split_map[lid] = "val"
    for lid in test_lesions["lesion_id"]:  split_map[lid] = "test"

    # ── 3. RESTRUCTURE ───────────────────────────────────────────────────────
    if PROCESSED_DIR.exists():
        print(f"Cleaning existing directory: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)

    stats = {split: {dx: 0 for dx in CLASS_NAMES} for split in ["train", "val", "test"]}
    corrupted_count = 0

    print("\nCopying images to structured directories...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        split = split_map.get(row["lesion_id"])
        if not split: continue # Should not happen

        dx = row["dx"]
        dest_dir = PROCESSED_DIR / split / dx
        dest_dir.mkdir(parents=True, exist_ok=True)

        src_path = find_image(row["image_id"], RAW_DIR)
        if src_path:
            # Optional: Basic corruption check on copy
            try:
                shutil.copy2(src_path, dest_dir / f"{row['image_id']}.jpg")
                stats[split][dx] += 1
            except Exception:
                corrupted_count += 1
        else:
            corrupted_count += 1

    # ── 4. REPORTING ──────────────────────────────────────────────────────────
    summary = {
        "total_images": len(df),
        "unique_lesions": len(lesions_df),
        "corrupted_or_missing": corrupted_count,
        "splits": stats
    }

    with open(STATS_PATH, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*50)
    print("DATASET QUALITY RESTRUCTURE COMPLETE")
    print("="*50)
    print(f"Report saved to: {STATS_PATH}")
    print(f"Missing/Corrupted: {corrupted_count}")
    
    for split, counts in stats.items():
        total = sum(counts.values())
        print(f"\n[{split.upper()}] Total: {total}")
        for dx, count in counts.items():
            print(f"  - {dx:<6}: {count}")

if __name__ == "__main__":
    main()
