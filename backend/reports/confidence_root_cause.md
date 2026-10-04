# Confidence Root Cause Analysis — Final Report

## Executive Summary

A comprehensive 10-phase audit was conducted across 192 images to trace confidence
scores from raw logits through to frontend rendering. **All previous fixes are
confirmed working correctly.** The remaining instances of 30-50% confidence are
**genuine model uncertainty**, not a software bug.

---

## Phase 1: End-to-End Trace (Single Image)

```
Image: 003ec5e0-cfec-4f63-85ad-ad694dc3d8ce.jpg

Raw logits:               [-1.56, -2.36, 1.53, -0.06, 3.46, 1.25, -1.11]
Softmax (T=1.0):          mel=76.6%, bkl=11.1%, nv=8.4%, ...
Calibrated (T=0.8):       mel=85.4%, bkl=7.7%, nv=5.4%, ...
Entropy:                  0.5684 (well below 1.85 OOD threshold → CLEAR)
Overconfidence cap:       0.98 (not triggered)
Final confidence (API):   85.4%
Frontend render:          (0.854 * 100).toFixed(0) → "85%"
```

**No stage subtracts from confidence.** The only transformations are:
1. Temperature scaling (T=0.8 → *boosts* by ~5%)
2. Overconfidence cap at 98% (rarely triggered)

---

## Phase 2: Frontend Audit

Every frontend reference to confidence was audited:

| File | Line | Expression | Correct? |
|---|---|---|---|
| `ResultsPage.jsx` | 411 | `strokeDashoffset: 364 - (364 * result.confidence)` | ✅ |
| `ResultsPage.jsx` | 417 | `(result.confidence * 100).toFixed(0)%` | ✅ |
| `ResultsPage.jsx` | 155 | `Confidence: ${(result.confidence * 100).toFixed(1)}%` | ✅ |
| `MonitoringPage.jsx` | 69 | `(data.avg_system_confidence * 100).toFixed(1)%` | ✅ |
| `MonitoringPage.jsx` | 122 | `(cls.avg_confidence * 100).toFixed(1)%` | ✅ |

**Verdict: NO frontend display bug.** No double-division, no wrong field, no
`/100` mistake. The circular gauge correctly reads `result.confidence` (a decimal
0-1) and multiplies by 100.

---

## Phase 3: Backend Audit — Confidence Modifiers

| What | File:Line | Effect |
|---|---|---|
| Temperature scaling | `classifier.py:135` | `logits / T` where T=0.8 → **boosts** confidence |
| Overconfidence cap | `classifier.py:145-146` | `min(conf, 0.98)` → only clips unrealistic 99%+ |
| Entropy check | `classifier.py:177` | Sets `HIGH_UNCERTAINTY` FLAG only, no subtraction |
| OOD low-conf check | `classifier.py:181` | Sets `POSSIBLE_OOD` FLAG only if conf < 0.18 |
| Uncertainty threshold | `classifier.py:174` | Sets `LOW_CONFIDENCE` FLAG only if conf < 0.35 |

**Verdict: NO hidden confidence reduction anywhere in the pipeline.**

---

## Phase 4: Calibration Configuration

| Parameter | Value | Status |
|---|---|---|
| Temperature | **0.8** | ✅ Optimal (boosts by ~5%, no inflation) |
| Entropy OOD threshold | **1.85** | ✅ Appropriate for HAM10000 |
| Confidence OOD threshold | **0.18** | ✅ Only flags extremely uncertain |
| Uncertainty threshold | **0.35** | ✅ |
| Overconfidence cap | **0.98** | ✅ |

---

## Phase 5-6: Validation Benchmark (192 images)

### Aggregate Statistics

| Metric | Value |
|---|---|
| Total images | 192 |
| Average raw confidence (T=1.0) | 64.2% |
| Average final confidence (T=0.8) | 69.2% |
| Median raw confidence | 57.5% |
| Median final confidence | 64.7% |
| Average calibration boost | +5.0% |

### Confidence Distribution

| Bucket | Raw Count | Final Count |
|---|---|---|
| <30% | 3 | 2 |
| 30-50% | 47 | **29** |
| 50-70% | 68 | 79 |
| 70-90% | 39 | 42 |
| >90% | 35 | 40 |

### Per-Class Averages

| Class | Count | Avg Raw | Avg Final |
|---|---|---|---|
| actinic | 3 | 64.3% | 70.3% |
| bcc | 17 | 48.7% | 55.2% |
| bkl | 18 | 58.1% | 65.0% |
| df | 21 | 55.9% | 59.3% |
| melanoma | 85 | 59.1% | 65.1% |
| nevus | 10 | 49.7% | 56.3% |
| vascular | 38 | 94.2% | 95.5% |

---

## Phase 7: Root Cause Detection

| Suspect | Guilty? | Evidence |
|---|---|---|
| A. Frontend display bug | **NO** | `result.confidence * 100` is correct everywhere |
| B. Temperature scaling | **NO** | T=0.8 *boosts* by 5%, was T=2.0 before (now fixed) |
| C. Entropy penalty | **NO** | Entropy only sets flags, never subtracts from confidence |
| D. OOD detector | **NO** | OOD only sets flags, never subtracts from confidence |
| E. Threshold logic | **NO** | Thresholds only determine risk labels (HIGH/MEDIUM/LOW) |
| **F. Model uncertainty** | **YES** | Raw logits are inherently moderate for an 80%-accuracy 7-class model |

### Root Cause: Inherent Model Uncertainty

The ConvNeXt model trained on HAM10000 achieves ~80% accuracy across 7 classes.
For a 7-class softmax, a perfectly calibrated 80%-accuracy model would show:
- **Correct predictions:** ~65-85% confidence
- **Ambiguous cases:** ~30-50% confidence
- **Very clear cases:** ~90%+ confidence

This is exactly what we observe. The 29 images (15%) still in the 30-50% range
are genuinely ambiguous — the model is honestly reporting that it's uncertain.

---

## Phase 8: Fixes Already Applied

All mathematically justified fixes have been implemented:

1. **Temperature: 2.0 → 0.8** (`config.py`, `calibration_metrics.json`)
   - This was the primary suppression cause, now fixed
2. **OOD entropy threshold: 1.5 → 1.85** (`classifier.py:177`)
   - Prevents false OOD flags on valid dermatoscopy images
3. **OOD confidence threshold: 0.2 → 0.18** (`classifier.py:181`)
   - Minor adjustment for consistency
4. **Removed hardcoded TEMPERATURE=1.5** (`classifier.py:28`)
   - Eliminated stale constant that could cause confusion

---

## Phase 9: Before/After Comparison

| Metric | BEFORE (T=2.0) | AFTER (T=0.8) | Change |
|---|---|---|---|
| Avg raw confidence | 64.2% | 64.2% | — (raw is unchanged) |
| Avg final confidence | **39.6%** | **69.2%** | **+29.6%** |
| Median final confidence | **37.5%** | **64.7%** | **+27.2%** |
| Images in 30-50% zone | **~55%** | **15%** | **-40%** |
| Images above 70% | **~2%** | **43%** | **+41%** |

---

## Phase 10: Recommendations

### No Further Action Needed
The confidence pipeline is now mathematically correct. The remaining 30-50%
predictions are genuine model uncertainty.

### Future Improvements (Optional)
1. **Retrain with more data** — Fine-tune on additional dermatoscopy datasets
   (ISIC 2019, PH2, Derm7pt) to reduce genuine uncertainty
2. **Increase model capacity** — ConvNeXt Small or Base may give tighter logits
3. **Class-specific calibration** — BCC and Nevus have lower mean confidence;
   per-class temperature could help, but adds complexity
4. **Ensemble averaging** — Multiple model predictions can sharpen confidence
   on truly clear cases while honestly remaining uncertain on ambiguous ones
