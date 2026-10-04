import os
from pathlib import Path
from tqdm import tqdm
from app.core.segmenter import segmenter
from PIL import Image

def preprocess_set(input_dir, output_dir):
    """Runs segmentation and cropping on a dataset directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all subdirectories (classes)
    classes = [d.name for d in input_path.iterdir() if d.is_dir()]
    
    for cls in classes:
        cls_input = input_path / cls
        cls_output = output_path / cls
        cls_output.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing class: {cls}")
        images = list(cls_input.glob("*.jpg"))
        
        for img_p in tqdm(images):
            try:
                # Segment and crop
                # segment_and_crop saves the output file, we just need to specify where
                target_output = cls_output / img_p.name
                if target_output.exists():
                    continue
                    
                cropped_path, _ = segmenter.segment_and_crop(str(img_p), output_path=str(target_output))
            except Exception as e:
                print(f"Error processing {img_p}: {e}")

if __name__ == "__main__":
    BASE = Path("C:/Users/Lenovo/medical-assistant-working/backend/data/processed")
    OUTPUT = Path("C:/Users/Lenovo/medical-assistant-working/backend/data/processed_segmented")
    
    print("Starting Offline Dataset Segmentation (Phase 5)...")
    
    # Process Train and Val
    for split in ["train", "val", "test"]:
        print(f"\n--- Processing {split} set ---")
        preprocess_set(BASE / split, OUTPUT / split)
        
    print("\nPhase 5: Offline Segmentation Complete.")
