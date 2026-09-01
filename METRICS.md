# Personal Parameters

| Field | Value |
| --- | --- |
| SID4 | 9486 |
| SEED | 9486 |
| SLICE | 486 |
| HP_ID | 0 |
| CLS_A | 6 |
| CLS_B | 0 |

# Baseline Configuration

```python
{
    "hidden_layers": [64, 32],
    "learning_rate": 0.001,
    "epochs": 30
}
```

# Modified Configuration

```python
{
    "hidden_layers": [32],
    "learning_rate": 0.001,
    "epochs": 30
}
```

# Training Seeds

- 9486
- 9487
- 9488

# Dataset Inspection

The headerless CSV is the transformed, approximately `[-1, 1]`-scaled dataset described in the assignment notes. Semantic names are assigned in the notebook using the matched schema. `SourceLabel=0` means diabetes and `SourceLabel=1` means no diabetes; the modeling target is `Diabetes = 1 - SourceLabel`.

| Metric | Value |
| --- | ---: |
| Rows | 759 |
| Source columns | 9 |
| Input features | 8 |
| Source label column | SourceLabel |
| Final target column | Diabetes |
| Target classes | 0, 1 |
| Duplicate rows | 0 |
| Total explicit missing values | 0 |

## Source and Final Target Distribution

| Label | Meaning | Count | Percentage |
| --- | --- | ---: | ---: |
| SourceLabel 0 | diabetes | 263 | 34.6509% |
| SourceLabel 1 | no diabetes | 496 | 65.3491% |
| Diabetes 0 | no diabetes | 496 | 65.3491% |
| Diabetes 1 | diabetes | 263 | 34.6509% |

## Zero-Value Treatment

Valid zeros are preserved in Pregnancies, DiabetesPedigreeFunction, and Age. SourceLabel is a valid binary label. Zero is treated as a missing-value sentinel only in the five medical measurement columns below.

| Column | Zero count |
| --- | ---: |
| Pregnancies | 111 |
| Glucose sentinel | 5 |
| BloodPressure sentinel | 35 |
| SkinThickness sentinel | 224 |
| Insulin sentinel | 371 |
| BMI sentinel | 11 |
| DiabetesPedigreeFunction | 1 |
| Age | 63 |
| SourceLabel | 263 |

## Fixed Split and Preprocessing

| Split | Rows | Class 0 count | Class 1 count | Class 0 percentage | Class 1 percentage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 531 | 347 | 184 | 65.3484% | 34.6516% |
| Validation | 114 | 75 | 39 | 65.7895% | 34.2105% |
| Testing | 114 | 74 | 40 | 64.9123% | 35.0877% |

- Split uses row indices, `random_state=9486`, and stratification.
- Train, validation, and test indices have no overlap and cover all 759 rows.
- Median imputation and `StandardScaler` are fit on training data only.
- Final processed feature and target arrays are `float32` with no remaining NaNs.

## Exploratory Data Analysis

EDA uses the full dataset descriptively, with zero sentinels masked only in the five affected medical measurement columns. Correlations use pairwise-complete observations.

### Target Class Distribution

| Diabetes class | Count | Percentage |
| ---: | ---: | ---: |
| 0 | 496 | 65.3491% |
| 1 | 263 | 34.6509% |

### Feature-to-Target Correlations

| Feature | Pearson correlation with Diabetes | Non-missing observations |
| --- | ---: | ---: |
| Glucose | 0.491538 | 754 |
| BMI | 0.315051 | 748 |
| Insulin | 0.303797 | 388 |
| SkinThickness | 0.262079 | 535 |
| Pregnancies | 0.218405 | 759 |
| BloodPressure | 0.169221 | 724 |
| DiabetesPedigreeFunction | 0.163246 | 759 |
| Age | 0.112565 | 759 |

### Generated Figures

- `figures/correlation_matrix.png` — 3197 × 2606 pixels
- `figures/feature_distributions.png` — 4767 × 2578 pixels

## PyTorch Experiment Settings

Status: Executed Colab results recorded

| Setting | Value |
| --- | --- |
| Baseline hidden layers | `[64, 32]` |
| Modified hidden layers | `[32]` |
| Learning rate | `0.001` |
| Epochs | `30` |
| Batch size | `32` |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Activation | ReLU |
| Output | One raw logit |
| Device | CPU |
| Training seeds | `9486`, `9487`, `9488` |
| Evaluation metric | Test accuracy, sigmoid threshold `0.5` |
| Accuracy variability | Sample standard deviation, `ddof=1` |
| Split | Existing fixed stratified 70/15/15 split |

### PyTorch Per-Run Results

| Framework | Model | Seed | Hidden Layers | Learning Rate | Epochs | Final Train Loss | Final Validation Loss | Test Accuracy |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | Baseline | 9486 | `[64, 32]` | 0.001 | 30 | 0.4061901019 | 0.5196852684 | 0.7982456088 |
| PyTorch | Baseline | 9487 | `[64, 32]` | 0.001 | 30 | 0.3937149304 | 0.5209291243 | 0.7894737124 |
| PyTorch | Baseline | 9488 | `[64, 32]` | 0.001 | 30 | 0.4080526210 | 0.5359846531 | 0.7894737124 |
| PyTorch | Modified | 9486 | `[32]` | 0.001 | 30 | 0.4490796123 | 0.5211900054 | 0.8157894611 |
| PyTorch | Modified | 9487 | `[32]` | 0.001 | 30 | 0.4427113002 | 0.5094487134 | 0.7982456088 |
| PyTorch | Modified | 9488 | `[32]` | 0.001 | 30 | 0.4452180164 | 0.5196500316 | 0.7982456088 |

### PyTorch Summary Results

| Model | Number of Runs | Mean Test Accuracy | Sample Std. Dev. | Minimum Test Accuracy | Maximum Test Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3 | 0.7923976779 (79.2398%) | 0.0050644567 (0.5064 pp) | 0.7894737124 | 0.7982456088 |
| Modified | 3 | 0.8040935596 (80.4094%) | 0.0101289479 (1.0129 pp) | 0.7982456088 | 0.8157894611 |

### PyTorch Parameters and Curve Diagnostics

| Model | Trainable parameters | Minimum validation-loss epoch | Minimum validation loss | Final validation loss | Final training loss | Final validation-training gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2,689 | 7 | 0.5048945870 | 0.5196852684 | 0.4061901019 | 0.1134951665 |
| Modified | 321 | 14 | 0.5032110570 | 0.5211900054 | 0.4490796123 | 0.0721103931 |

### PyTorch Checkpoints

- `checkpoints/pytorch/pytorch_baseline_seed_9486.pt`
- `checkpoints/pytorch/pytorch_baseline_seed_9487.pt`
- `checkpoints/pytorch/pytorch_baseline_seed_9488.pt`
- `checkpoints/pytorch/pytorch_modified_seed_9486.pt`
- `checkpoints/pytorch/pytorch_modified_seed_9487.pt`
- `checkpoints/pytorch/pytorch_modified_seed_9488.pt`

Loss figure: `figures/pytorch_loss_curves.png` (`2970 × 1765` pixels, `277891` bytes).

## TensorFlow/Keras Experiment Settings

Status: Executed Colab results recorded

| Setting | Value |
| --- | --- |
| Baseline hidden layers | `[64, 32]` |
| Modified hidden layers | `[32]` |
| Learning rate | `0.001` |
| Epochs | `30` |
| Batch size | `32` |
| Optimizer | Adam |
| Loss | `binary_crossentropy` |
| Metrics | `accuracy` |
| Activation | ReLU hidden layers; sigmoid output |
| Device | CPU |
| Training seeds | `9486`, `9487`, `9488` |
| Evaluation metric | `model.evaluate()` accuracy; manual `model.predict()` threshold `0.5` check |
| Accuracy variability | Sample standard deviation, `ddof=1` |
| Split and arrays | Existing fixed preprocessed split and arrays |

### TensorFlow Per-Run Results

| Framework | Model | Seed | Hidden Layers | Learning Rate | Epochs | Final Train Loss | Final Validation Loss | Test Accuracy | Test Loss |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TensorFlow | Baseline | 9486 | `[64, 32]` | 0.001 | 30 | 0.3907621503 | 0.5301329494 | 0.7719298005 | 0.4734252095 |
| TensorFlow | Baseline | 9487 | `[64, 32]` | 0.001 | 30 | 0.3850820661 | 0.5131751895 | 0.7807017565 | 0.4801962972 |
| TensorFlow | Baseline | 9488 | `[64, 32]` | 0.001 | 30 | 0.3829312623 | 0.5114467740 | 0.8333333135 | 0.4625320733 |
| TensorFlow | Modified | 9486 | `[32]` | 0.001 | 30 | 0.4514864385 | 0.5250613689 | 0.7982456088 | 0.4339624047 |
| TensorFlow | Modified | 9487 | `[32]` | 0.001 | 30 | 0.4461356103 | 0.5085556507 | 0.8157894611 | 0.4465323389 |
| TensorFlow | Modified | 9488 | `[32]` | 0.001 | 30 | 0.4451223900 | 0.5071377158 | 0.8070175648 | 0.4554108083 |

### TensorFlow Summary Results

| Model | Number of Runs | Mean Accuracy | Sample Std. Dev. | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 3 | 0.7953216235 (79.5322%) | 0.033210 (3.3210 pp) | 0.7719298005 | 0.8333333135 |
| Modified | 3 | 0.8070175449 (80.7018%) | 0.008772 (0.8772 pp) | 0.7982456088 | 0.8157894611 |

Mean improvement: approximately 1.17 percentage points. Accuracy variability uses sample standard deviation with `ddof=1`.

### TensorFlow Parameters and Curve Diagnostics

| Model | Trainable parameters | Minimum validation-loss epoch | Minimum validation loss | Final validation loss | Final training loss | Final validation-training gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2,689 | 20 | 0.5240626335 | 0.5301329494 | 0.3907621503 | 0.1393707991 |
| Modified | 321 | 30 | 0.5250613689 | 0.5250613689 | 0.4514864385 | 0.0735749304 |

### TensorFlow Checkpoints

- `checkpoints/tensorflow/tensorflow_baseline_seed_9486.keras`
- `checkpoints/tensorflow/tensorflow_baseline_seed_9487.keras`
- `checkpoints/tensorflow/tensorflow_baseline_seed_9488.keras`
- `checkpoints/tensorflow/tensorflow_modified_seed_9486.keras`
- `checkpoints/tensorflow/tensorflow_modified_seed_9487.keras`
- `checkpoints/tensorflow/tensorflow_modified_seed_9488.keras`

Loss figure: `figures/tensorflow_loss_curves.png` (`2970 × 1765` pixels, `263506` bytes).

# Neural Network Results

Status: PyTorch and TensorFlow results recorded

## Baseline Results

Recorded in the PyTorch results tables above.

## Modified Results

Recorded in the PyTorch results tables above.

## Final Neural-Network Comparison

| Framework | Model | Parameters | Mean accuracy | Sample SD | Min | Max | Modified improvement |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | Baseline | 2,689 | 0.7923976779 | 0.0050644567 | 0.7894737124 | 0.7982456088 | — |
| PyTorch | Modified | 321 | 0.8040935596 | 0.0101289479 | 0.7982456088 | 0.8157894611 | 0.0116958817 |
| TensorFlow | Baseline | 2,689 | 0.7953216235 | 0.033210 | 0.7719298005 | 0.8333333135 | — |
| TensorFlow | Modified | 321 | 0.8070175449 | 0.008772 | 0.7982456088 | 0.8157894611 | 0.0116959214 |

Parameter reduction: 2,689 to 321, or 2,368 fewer parameters and approximately 88.1% fewer. Both frameworks used the same fixed split, preprocessing, processed arrays, and test accuracy metric. Sample standard deviation uses `ddof=1`.

# CUDA Results

Status: Pending

## Compilation

Pending

## Timing

Pending

## Profiler

Pending
