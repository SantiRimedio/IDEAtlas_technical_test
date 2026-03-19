# Model Architecture

## MBCNN (Multi-Branch Convolutional Neural Network)

Defined in `src/training/mbcnn.py` as `mbcnn()`.

### Structure

```
Input Branch 1 (S2: 128×128×6)  ──→ Conv2D(16) ──┐
Input Branch 2 (DEN: 128×128×N) ──→ Conv2D(16) ──┤
                                                   ├──→ Concatenate
                                                   │
                                            Encoder (U-Net style)
                                    ┌───────────────────────────┐
                                    │ e0: 2× Conv(16) + BN      │
                                    │ e1: MaxPool → 2× Conv(32) │
                                    │ e2: MaxPool → 2× Conv(64) │
                                    └───────────────────────────┘
                                                   │
                                            Decoder (skip connections)
                                    ┌───────────────────────────┐
                                    │ d2: UpSample + skip(e1)    │
                                    │     2× ConvTranspose(32)   │
                                    │ d1: UpSample + skip(e0)    │
                                    │     2× ConvTranspose(16)   │
                                    └───────────────────────────┘
                                                   │
                                    Conv2D(N_CLASSES, 1×1) → Softmax
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CL` | 3 | Number of output classes |
| `input_shapes` | dict | Per-branch input shapes |
| `dropout_rate` | 0.2 | Dropout probability |
| `batch_norm` | False | Enable batch normalization |
| `nfilters` | [16, 32, 64] | Filters per encoder level |

## MTCNN (Multi-Task CNN)

Defined in `src/training/mbcnn.py` as `mtcnn()`. Same encoder-decoder structure but with two output heads:

1. **Regression head**: `sigmoid` activation → density prediction (1 channel)
2. **Classification head**: Concatenates decoder output with regression output → `softmax` (N classes)

Filter sizes: [8, 16, 32] (reduced from MBCNN).

## Loss Functions

Defined in `src/training/losses.py`.

| Loss | Parameters | Use |
|------|-----------|-----|
| `FocalLoss` | gamma=2.0, alphas=0.25 | Class imbalance handling |
| `DiceLoss` | class_idx | Per-class spatial overlap |
| `CombinedDiceFocalLoss` | dice_weight=1.0, focal_weight=2.0 | Default training loss |

## Training Configuration

From `notebooks/06_train.ipynb`:
- Optimizer: Adam, lr=1e-4
- Patch size: 128×128
- N_CLASSES: 3
- Loss: CombinedDiceFocalLoss (target class index 2 = slum)
- Class weights computed from training set distribution

## Inference

Defined in `src/inference/inference.py` as `full_inference_mbcnn()`:
- Sliding window with stride = patch_size // 2
- Hann window blending for smooth transitions
- Batch processing (default batch_size=32)
- Output clipped to AOI boundary vector
