import sys
from pathlib import Path
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from app.core.classifier import classifier
from torchvision import transforms
from PIL import Image

def main():
    model = classifier.model
    if model is None:
        print("Model is not loaded.")
        return
        
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    
    upload_dir = Path('uploads')
    images = list(upload_dir.glob('*.jpg'))[:10] + list(upload_dir.glob('*.png'))[:10]
    
    if not images:
        print("No images found in uploads folder.")
        return

    print("="*60)
    print(f"{'Image Name':<40} | {'Raw Softmax Max (%)':<20}")
    print("="*60)
    
    for img_path in images:
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(classifier.device)
            with torch.no_grad():
                logits = model(tensor)
            
            # Raw softmax without temperature scaling (T=1.0)
            raw_probs = F.softmax(logits, dim=1)[0]
            raw_max = raw_probs.max().item() * 100
            
            print(f"{img_path.name:<40} | {raw_max:>6.2f}%")
        except Exception as e:
            print(f"{img_path.name:<40} | Error: {e}")

if __name__ == '__main__':
    main()
