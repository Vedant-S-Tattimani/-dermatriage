import pandas as pd
import shutil
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    images_dir = raw_dir / "HAM10000_images"
    metadata_path = raw_dir / "metadata.csv"
    
    # 5. Ensure: Create actinic folder if not exists
    target_dir = base_dir / "data" / "processed" / "actinic"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if not metadata_path.exists():
        print(f"Error: metadata.csv not found at {metadata_path}")
        return

    print("Loading metadata...")
    # 1. Load metadata.csv using pandas
    df = pd.read_csv(metadata_path)
    
    # 2. Filter rows: dx == "akiec"   # actinic keratosis
    actinic_df = df[df["dx"] == "akiec"]
    total_found = len(actinic_df)
    
    copied = 0
    missing = 0
    
    print(f"Extracting {total_found} actinic keratosis images...")
    
    # 3. For each filtered row: Get image_id, Construct filename
    for _, row in actinic_df.iterrows():
        image_id = row["image_id"]
        filename = f"{image_id}.jpg"
        
        source_path = images_dir / filename
        target_path = target_dir / filename
        
        # 4. Copy images from data/raw/HAM10000_images/ TO data/processed/actinic/
        # 5. Skip missing images safely
        if source_path.exists():
            shutil.copy2(source_path, target_path)
            copied += 1
        else:
            missing += 1
            
    # 6. Print: Total actinic images found, Total successfully copied, Missing images count
    print("-" * 30)
    print("Extraction Summary:")
    print(f"Total actinic images found: {total_found}")
    print(f"Total successfully copied: {copied}")
    print(f"Missing images count: {missing}")
    print("-" * 30)

if __name__ == "__main__":
    main()
