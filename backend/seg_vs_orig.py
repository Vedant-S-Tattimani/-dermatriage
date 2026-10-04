import sys, json
sys.path.insert(0, '.')
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from app.core.classifier import classifier
from app.core.segmenter import segmenter
from app.config import settings
from torchvision import transforms
from PIL import Image

def main():
    T = settings.CALIBRATION_TEMPERATURE
    print(f'Current Temperature: {T}')
    print()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    upload_dir = Path('uploads')
    # Filter ONLY original uploaded images (skip cropped_, mask_, xai_, gradcam_ prefixed)
    images = [p for p in upload_dir.glob('*.jpg') 
              if not p.name.startswith(('cropped_','mask_','xai_','gradcam_','val_'))][:20]

    print('='*95)
    header = f"{'Image':<42} | {'OrigClass':<10} | {'Orig%':>6} | {'SegClass':<10} | {'Seg%':>6} | {'Diff':>6}"
    print(header)
    print('='*95)

    model = classifier.model
    diffs = []
    class_changed = 0

    for img_path in images:
        try:
            # A: Direct inference on ORIGINAL image
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(classifier.device)
            with torch.no_grad():
                logits = model(tensor)
            cal_probs = F.softmax(logits / T, dim=1)[0]
            orig_conf = cal_probs.max().item()
            orig_class = settings.CLASS_NAMES[cal_probs.argmax().item()]
            
            # B: Inference on SEGMENTED + CROPPED image (what triage_service actually does)
            try:
                cropped_path, mask_path = segmenter.segment_and_crop(str(img_path))
                img2 = Image.open(cropped_path).convert('RGB')
                tensor2 = transform(img2).unsqueeze(0).to(classifier.device)
                with torch.no_grad():
                    logits2 = model(tensor2)
                cal_probs2 = F.softmax(logits2 / T, dim=1)[0]
                seg_conf = cal_probs2.max().item()
                seg_class = settings.CLASS_NAMES[cal_probs2.argmax().item()]
            except Exception as e:
                seg_conf = orig_conf
                seg_class = orig_class
            
            diff = (seg_conf - orig_conf) * 100
            diffs.append(diff)
            
            if seg_class != orig_class:
                class_changed += 1
            
            marker = ' <<<' if diff < -10 else ''
            print(f'{img_path.name:<42} | {orig_class:<10} | {orig_conf*100:>5.1f}% | {seg_class:<10} | {seg_conf*100:>5.1f}% | {diff:>+5.1f}%{marker}')
        except Exception as e:
            print(f'{img_path.name:<42} | ERROR: {e}')

    print('='*95)
    if diffs:
        print(f'Average confidence change from segmentation: {np.mean(diffs):+.1f}%')
        print(f'Median confidence change:                    {np.median(diffs):+.1f}%')
        worse = sum(1 for d in diffs if d < -5)
        print(f'Images where segmentation HURTS (>5% drop):  {worse}/{len(diffs)}')
        print(f'Images where class CHANGED after segmentation: {class_changed}/{len(diffs)}')

if __name__ == '__main__':
    main()
