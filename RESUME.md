# Guía de Retoma — Hyperparameter Tuning MBCNN AMBA
**Última actualización:** 2026-03-19
**Estado:** D9 completo, D10 interrumpido en época 8 (best checkpoint guardado), listo para continuar en GPU

---

## Estado del experimento

### Bloques A–C (grid pequeño, 178 patches train) — COMPLETOS

| Experimento | cls2_F1 | cls2_Recall | cls2_Precision | Conclusión |
|---|---|---|---|---|
| baseline_combined (A) | 0.000 | 0.000 | 0.000 | ❌ focal no ayuda |
| **pure_dice (A)** | **0.019** | **0.031** | **0.014** | ✅ mejor de A-C |
| pure_focal (A) | 0.000 | 0.000 | 0.000 | ❌ |
| dice_recall_monitor (B) | 0.003 | 0.042 | 0.002 | ❌ peor que pure_dice |
| dice_cls2_10x_raw (B) | 0.001 | 0.000 | 0.001 | ❌ raw weights inestables |
| dice_cls2_50x_raw (B) | 0.003 | 0.056 | 0.001 | ❌ |
| baseline_combined_den3 (C) | 0.000 | 0.000 | 0.000 | ❌ DEN3 no ayuda |
| pure_dice_den3 (C) | 0.005 | 0.022 | 0.003 | ❌ DEN3 peor que DEN1 |

**Conclusiones A–C:**
- Mejor config: `pure_dice` + monitor `val_class2_recall` + DEN 1-banda
- El cuello de botella principal era **escasez de datos** (178 patches), no la loss
- DEN3 (3-band multiscale density) → **RECHAZADO**, siempre peor

---

### Bloque D (grid grande, 5,221 train patches) — PARCIALMENTE COMPLETO

| Experimento | cls2_F1 | cls2_Recall | cls2_Precision | mIoU | Épocas | Estado |
|---|---|---|---|---|---|---|
| **large_pure_dice (D9)** | **0.224** | **0.735** | **0.132** | 0.342 | 13 | ✅ COMPLETO |
| large_pure_dice_oversample (D10) | — | — | — | — | 9/40 | ⚠️ INTERRUMPIDO |

**D10 parcial — val_class2_recall por época** (best weights guardados en `experiments/exp_02_large_pure_dice_oversample/best_model.weights.h5`):

| Época | val_cls2_recall |
|---|---|
| 0 | 0.469 |
| 1 | 0.711 |
| 2 | 0.758 |
| 3 | 0.747 |
| 4 | 0.810 |
| 5 | 0.840 |
| 6 | 0.784 |
| **7** | **0.854** ← best checkpoint |
| 8 | 0.836 |
| *killed* | — |

- LR bajó a 5e-5 en época 8 (ReduceLROnPlateau tras patience=6 sin mejora en val_loss)
- Tendencia positiva vs D9 (0.854 > 0.674 best val recall en D9)
- D10 **no llegó a evaluación en test**, no está en results.csv

**Conclusiones D:**
- **Volumen de datos fue el factor dominante**: cls2_F1 pasó de 0.019 → 0.224 (10×) solo con más datos
- Recall muy bueno (0.735), pero **precision baja (0.132)** → muchos falsos positivos
- Oversample 5× parece mejorar recall (val 0.854 vs 0.674) pero falta ver efecto en precision/F1 final
- Cada época del grid grande tarda ~5-7 min en CPU; en GPU debería ser ~5-10× más rápido

---

## Qué hacer en la GPU machine

### Paso 1 — Completar D10 (ya tiene datos y checkpoint)
```bash
# D10 fue interrumpida. Como ya tiene best_model.weights.h5,
# simplemente re-lanzar Block D — el runner saltea large_pure_dice (ya en results.csv)
# y re-corre large_pure_dice_oversample desde época 0.
# Nota: no hay resume de checkpoint en el runner actual, re-entrena desde cero.
python -m src.evaluation.experiment_runner --block d
```
Alternativa más rápida: solo correr D10 por nombre:
```bash
python -m src.evaluation.experiment_runner --run large_pure_dice_oversample
```

### Paso 2 — Bloque E (próximos experimentos recomendados)

El problema principal ahora es **precision baja (~0.13)**. El modelo recuerda bien las villas pero genera muchos falsos positivos. Experimentos sugeridos:

#### E11 — `large_dice_focal` (añadir focal para mejorar precision)
```python
ExperimentConfig(
    name="large_dice_focal",
    use_dice=True, use_focal=True,
    focal_alpha=0.85, focal_gamma=2.5, dice_focal_ratio=3.0,
    class2_weight_multiplier=1.0,
    monitor="val_class2_recall",
    epochs=40, use_large_grid=True, class2_oversample_factor=1,
    block="e",
    notes="Dice+focal en large grid — focal debería reducir falsos positivos"
)
```

#### E12 — `large_oversample_2x` (oversample más conservador)
```python
ExperimentConfig(
    name="large_oversample_2x",
    use_dice=True, use_focal=False,
    class2_weight_multiplier=1.0,
    monitor="val_class2_recall",
    epochs=40, use_large_grid=True, class2_oversample_factor=2,
    block="e",
    notes="Oversample 2x — menos agresivo que 5x, potencialmente mejor precision"
)
```

#### E13 — `large_dice_focal_oversample` (combinación)
```python
ExperimentConfig(
    name="large_dice_focal_oversample",
    use_dice=True, use_focal=True,
    focal_alpha=0.85, focal_gamma=2.5, dice_focal_ratio=3.0,
    class2_weight_multiplier=1.0,
    monitor="val_class2_recall",
    epochs=40, use_large_grid=True, class2_oversample_factor=5,
    block="e",
    notes="Dice+focal + 5x oversample en large grid"
)
```

#### E14 — `large_cls2_weight_boost` (boost de peso de clase 2)
```python
ExperimentConfig(
    name="large_cls2_weight_boost",
    use_dice=True, use_focal=False,
    class2_weight_multiplier=3.0,
    normalize_class_weights=True,
    monitor="val_class2_recall",
    epochs=40, use_large_grid=True, class2_oversample_factor=1,
    block="e",
    notes="Pure dice + 3x class2 weight multiplier en large grid (normalizado)"
)
```

**Orden de prioridad:** E11 > E12 > E13 > E14
Razón: E11 es la hipótesis más clara (focal loss penaliza más las predicciones confiadas erróneas → menos FP). E12 y E13 exploran el trade-off de oversample.

---

## Datasets

- **patches_large/**: ~2.8 GB, **gitignoreado** — hay que transferir manualmente
  ```bash
  rsync -avz --progress datasets/patches_large/ usuario@gpu:/ruta/Techo/datasets/patches_large/
  ```
- **patches/ (small)**: ~50 MB, también gitignoreado
- **data/boundaries/**: trackeados en git ✅
- **models/**: trackeados en git ✅
- **experiments/**: trackeados en git ✅

## Comandos clave

```bash
# Instalar dependencias
pip install -r requirements.txt

# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Ver resumen de resultados actuales
python -m src.evaluation.experiment_runner --summary

# Completar D10
python -m src.evaluation.experiment_runner --run large_pure_dice_oversample

# Correr Bloque E (una vez añadido al runner)
python -m src.evaluation.experiment_runner --block e
```

---

## Métricas objetivo (Bloque E)

| Métrica | D9 actual | Objetivo E |
|---|---|---|
| cls2_F1 | 0.224 | > 0.35 |
| cls2_Recall | 0.735 | > 0.70 (mantener) |
| cls2_Precision | 0.132 | > 0.30 |
| mIoU | 0.342 | > 0.38 |
