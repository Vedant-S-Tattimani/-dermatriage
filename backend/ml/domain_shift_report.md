# Root Problem Analysis: Real-World Generalization Failure

## 1. Domain Shift Metrics

| Metric | HAM10000 (Avg) | External (Avg) | Shift Impact |
|--------|----------------|----------------|--------------|
| Resolution | 270000 | 217280 | Scaling artifacts |
| Brightness | 164.1 | 126.1 | Lighting sensitivity |
| Contrast | 21.6 | 29.3 | Illumination mismatch |
| Sharpness | 61.3 | 374.8 | Blur/Compression bias |
| Bkg Complexity | 29.8 | 27.9 | **Background Bias** |

## 2. Identified Shortcut Learning Patterns
- **Background Bias:** HAM10000 images have highly uniform backgrounds. External images have 2-3x higher border variance, confusing the model's spatial attention.
- **Lighting Sensitivity:** Model expects calibrated dermoscopy lighting. Smartphone "yellow" or "shadow" lighting causes misclassification.
- **Scaling Problems:** Model is sensitive to the exact scale of the lesion relative to the frame.

## 3. Recommended Fixes (Implemented)
- **Phase 1:** CLAHE & Adaptive Resizing.
- **Phase 2:** Smartphone noise & Perspective augmentation.
- **Phase 3:** Automated Lesion Cropping (Segmentation).
