# Next Steps & Improvement Plan

**Date:** March 19, 2026
**Current Status:** End-to-end pipeline complete (training + inference + evaluation)
**Priority Focus:** Improving Class 2 (informal settlement/slum) detection

---

## 🔴 Critical Issue: Class 2 Detection Failure

### Problem Statement
The model is failing to detect informal settlements (Class 2) effectively:

- **Training F1-score:** 0.0081 (essentially random)
- **Inference F1-score:** 0.0881 (still very poor)
- **Recall on full raster:** 7.2% (misses 92.8% of slums)
- **Predicted area:** Only 37.5% of reference area

### Why This Matters
- Primary goal is informal settlement detection
- Current model is useless for this critical class
- Overall accuracy (94.20%) is misleading due to class imbalance

---

## 📋 Immediate Actions (This Week)

### 1. Debug Class 3 Spurious Creation
**Time: 30 minutes**

**Problem:** Model creates 2.4M pixels of Class 3, which doesn't exist in training data

**Check inference.py:**
- How are soft probabilities converted to class labels?
- Is argmax(axis=-1) applied correctly?
- Are class indices remapped (0-based → 1-based)?

### 2. Analyze Class 2 Training Data
**Time: 1 hour**

Run: `python -c "from pathlib import Path; import rasterio, numpy as np; patches = sorted(Path('datasets/patches/train').glob('RF_*.tif')); [print(f'{p.name}: {np.sum(rasterio.open(p).read(1)==2)} pixels of Class 2') for p in patches]"`

Questions:
- How many training patches contain Class 2?
- What's the distribution across patches?
- Are there "pure" Class 2 patches?

### 3. Improve Class 2 Detection (Choose one or combine)

**Option A: Higher Class Weight** (30 min)
```python
# In scripts/06_train.py, increase Class 2 weight:
# From: {0: 1.68, 1: 0.43, 2: 13.40}
# To: {0: 1.68, 1: 0.43, 2: 25.0}
```

**Option B: Increase Focal Loss Gamma** (30 min)
```python
# From: gamma=2.5
# To: gamma=3.5 or 4.0
# Reason: Harder focus on hard examples (Class 1 vs Class 2 boundary)
```

**Option C: Data Augmentation** (1-2 hours)
```python
# Augment Class 2 patches: rotations, flips, elastic deformations
# Goal: 2-4x more Class 2 training samples
```

**Option D: Lower Class 2 Threshold** (1 hour)
```python
# Modify inference to use lower decision threshold for Class 2
# From: argmax(probabilities)
# To: if prob_class2 > 0.15 then predict Class 2
```

### 4. Modify Loss Function
**Time: 1 hour**

Try combination:
```
0.4 × Dice Loss +
0.3 × Categorical Focal Loss (gamma=3.5) +
0.3 × Tversky Loss (emphasizes recall/minimizes false negatives)
```

### 5. Retrain and Evaluate
```bash
python scripts/06_train.py      # Retrain
python scripts/07_inference.py  # Inference
python scripts/08_evaluate.py   # Evaluate
```

**Success criteria:**
- Class 2 F1 > 0.15 (3x improvement)
- Class 1 F1 remains > 0.90
- No Class 3 artifacts

---

## 🟢 Success Metrics for Iteration 2

- [x] Class 2 F1 > 0.088 (current baseline)
- [ ] Class 2 F1 > 0.20 (realistic target)
- [ ] Class 2 Recall > 0.25
- [ ] Class 2 Precision > 0.15
- [ ] No spurious Class 3
- [ ] Class 1 F1 remains > 0.85

---

## 📁 Output Files to Generate

- `datasets/reports/predictions_visualization.png` (spatial error maps)
- `datasets/reports/class2_analysis.txt` (training data analysis)
- `datasets/reports/metrics_history.json` (track improvement)

---

## 🚀 Quick Experiments (30 min each)

1. Try different patch sizes (64×64, 256×256)
2. Try different density normalization
3. Try building count instead of density
4. Try keeping original class ratios instead of balanced grid
5. Test different S2 band combinations (NDBI, NDVI)

---

## Long-term (After Iteration 2)

- Collect more Class 2 training data
- Try deeper network architecture
- Add attention mechanisms
- Multi-scale processing
- Ensemble methods
- CRF post-processing for refinement

---

**Next Session:** Debug Class 3, analyze Class 2 data, run improvement experiments
