import sys
from pathlib import Path
from torchvision import datasets

def main():
    base_dir = Path(__file__).resolve().parent.parent
    processed_dir = base_dir / "data" / "processed"

    # 1. Ensure directory structure
    melanoma_dir = processed_dir / "melanoma"
    benign_dir = processed_dir / "benign"
    actinic_dir = processed_dir / "actinic"

    melanoma_dir.mkdir(parents=True, exist_ok=True)
    benign_dir.mkdir(parents=True, exist_ok=True)
    actinic_dir.mkdir(parents=True, exist_ok=True)

    # 2. Count number of images in each folder
    mel_count = len(list(melanoma_dir.glob("*.jpg")))
    benign_count = len(list(benign_dir.glob("*.jpg")))
    actinic_count = len(list(actinic_dir.glob("*.jpg")))

    # 3. Print distribution clearly
    print("--------------------------------------------------")
    print("DATASET DISTRIBUTION (3-Class Classification)")
    print("--------------------------------------------------")
    print(f"Melanoma (mel):     {mel_count} images")
    print(f"Benign (nv):        {benign_count} images")
    print(f"Actinic (akiec):    {actinic_count} images")
    print("--------------------------------------------------")

    # 4. Warn if any class has < 50 images
    warnings = 0
    for name, count in [("melanoma", mel_count), ("benign", benign_count), ("actinic", actinic_count)]:
        if count < 50:
            print(f"WARNING: Class '{name}' has only {count} images (less than 50).")
            warnings += 1

    if warnings > 0:
        print(f"Total Warnings: {warnings}")

    # 5. Confirm ImageFolder can read dataset without errors
    try:
        dataset = datasets.ImageFolder(root=str(processed_dir))
        print(f"\nImageFolder successfully read dataset.")
        print(f"Detected classes: {dataset.classes}")
        print(f"Class to index mapping: {dataset.class_to_idx}")
    except Exception as e:
        print(f"\nERROR: ImageFolder failed to read dataset.")
        print(f"Exception: {e}")
        print("Validation Status: FAILED")
        sys.exit(1)

    # 6. Output validation status
    if actinic_count == 0:
        print("\nERROR: Actinic folder is missing or empty.")
        print("Validation Status: FAILED")
        sys.exit(1)

    print("\nValidation Status: PASSED")

if __name__ == "__main__":
    main()
