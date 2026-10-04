import sys, json, time
from pathlib import Path
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import numpy as np
from app.core.classifier import classifier
from app.config import settings
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
    images = list(upload_dir.glob('*.jpg'))[:100]
    if not images:
        print("No images found.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Configuration
    # ─────────────────────────────────────────────────────────────────────────
    print("================================================")
    print("STEP 4: CALIBRATION CONFIGURATION")
    print("================================================")
    print(f"Temperature:            {settings.CALIBRATION_TEMPERATURE}")
    print(f"OOD Entropy Threshold:  1.85 (Hardcoded in classifier.py)")
    print(f"Confidence Threshold:   {settings.PREVENT_OVERCONFIDENCE_CAP} (Cap)")
    print(f"Uncertain Threshold:    {settings.UNCERTAINTY_THRESHOLD}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Trace One Prediction End-to-End
    # ─────────────────────────────────────────────────────────────────────────
    img_path = images[0]
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(classifier.device)
    
    print("================================================")
    print("STEP 1: END-TO-END TRACE")
    print("================================================")
    print(f"Image: {img_path.name}")
    
    with torch.no_grad():
        logits = model(tensor)
    
    print(f"Raw logits:                 {logits[0].cpu().numpy()}")
    
    raw_probs = F.softmax(logits, dim=1)[0]
    print(f"Softmax probabilities:      {raw_probs.cpu().numpy()}")
    
    top3_vals, top3_idx = torch.topk(raw_probs, 3)
    top3_classes = [settings.CLASS_NAMES[i] for i in top3_idx.cpu().numpy()]
    print(f"Top-3 classes:              {top3_classes}")
    print(f"Top-3 raw probabilities:    {top3_vals.cpu().numpy()}")
    
    # Calibration
    T = settings.CALIBRATION_TEMPERATURE
    scaled_logits = logits / T
    cal_probs = F.softmax(scaled_logits, dim=1)[0]
    cal_conf = cal_probs.max().item()
    print(f"Confidence after calibration: {cal_conf:.4f}")
    
    # Entropy penalty / OOD
    entropy = -torch.sum(cal_probs * torch.log(cal_probs + 1e-10)).item()
    print(f"Entropy value:              {entropy:.4f}")
    
    final_conf = min(cal_conf, settings.PREVENT_OVERCONFIDENCE_CAP)
    print(f"Confidence after entropy:     {final_conf:.4f} (Entropy doesn't subtract directly in classifier.py)")
    print(f"Confidence after OOD:         {final_conf:.4f} (OOD flags uncertainty, doesn't subtract)")
    print(f"Confidence returned by API:   {final_conf:.4f}")
    print(f"Frontend math logic:        (confidence * 100).toFixed(0) -> {(final_conf * 100):.0f}%")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 & 6: Run 100 Validation Images
    # ─────────────────────────────────────────────────────────────────────────
    print("================================================")
    print("STEP 5 & 6: VALIDATION & STATISTICS (100 IMAGES)")
    print("================================================")
    
    stats = []
    for img_path in images:
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(classifier.device)
            with torch.no_grad():
                logits = model(tensor)
            
            # Raw
            raw_p = F.softmax(logits, dim=1)[0]
            raw_c = raw_p.max().item()
            raw_cls = settings.CLASS_NAMES[raw_p.argmax().item()]
            
            # Final
            scaled = logits / settings.CALIBRATION_TEMPERATURE
            cal_p = F.softmax(scaled, dim=1)[0]
            cal_c = min(cal_p.max().item(), settings.PREVENT_OVERCONFIDENCE_CAP)
            cal_cls = settings.CLASS_NAMES[cal_p.argmax().item()]
            
            # Pseudo 'correct' logic: assume model is correct for statistics gathering 
            # if we don't have true labels, or just mark all as "model's prediction".
            # For pure math auditing, we only care about raw vs final.
            stats.append({
                'raw': raw_c,
                'final': cal_c,
                'cls': cal_cls
            })
        except Exception as e:
            pass

    raw_arr = np.array([s['raw'] for s in stats])
    final_arr = np.array([s['final'] for s in stats])
    
    print(f"Average raw confidence:        {raw_arr.mean():.4f}")
    print(f"Average calibrated confidence: {final_arr.mean():.4f}")
    print(f"Average reduction:             {raw_arr.mean() - final_arr.mean():.4f}")

    if (raw_arr.mean() - final_arr.mean()) > 0.20:
        print("\n================================================")
        print("STEP 7: ROOT CAUSE ALERT (Reduction > 20%)")
        print("================================================")
        print("Wait! We changed Temperature to 0.8 earlier! Let's see if it took effect.")
        print("If reduction is large, T must still be > 1.0.")

if __name__ == '__main__':
    main()
