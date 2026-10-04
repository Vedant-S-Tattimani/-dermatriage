"""
Utility to visualize and save previews of the augmented training samples.
Ensures that the augmentation pipeline is working as expected.
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ml.data_loader import get_dataloaders, MEAN, STD

def denormalize(tensor):
    """Reverses ImageNet normalization for visualization."""
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    return tensor

def main():
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "ml" / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataloaders...")
    train_loader, _, _, _ = get_dataloaders(batch_size=8)
    
    # Get a single batch
    images, labels = next(iter(train_loader))
    
    # Class names for labeling (ImageFolder order)
    class_names = list(train_loader.dataset.class_to_idx.keys())
    
    print(f"Generating {len(images)} previews...")
    
    combined_images = []
    
    for i in range(len(images)):
        img_tensor = images[i].clone()
        img_denorm = denormalize(img_tensor)
        
        # Convert to PIL for saving
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        class_label = class_names[labels[i].item()]
        save_path = output_dir / f"aug_preview_{i}_{class_label}.jpg"
        img_pil.save(save_path)
        combined_images.append(img_np)
        print(f"  Saved: {save_path.name}")

    # Optional: Create a grid if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 8))
        for i in range(len(combined_images)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(combined_images[i])
            plt.title(class_names[labels[i].item()])
            plt.axis('off')
        
        grid_path = output_dir / "augmentation_grid.png"
        plt.tight_layout()
        plt.savefig(grid_path)
        print(f"\nGrid visualization saved to: {grid_path}")
    except ImportError:
        print("\n[Note] Matplotlib not found. Skipping grid generation.")

    print("\nAugmentation check complete. Files are in 'ml/previews/'")

if __name__ == "__main__":
    main()
