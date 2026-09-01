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

Status: Pending Colab execution

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

Pending Colab execution.

| Framework | Model | Seed | Hidden Layers | Learning Rate | Epochs | Final Train Loss | Final Validation Loss | Test Accuracy |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

### PyTorch Summary Results

Pending Colab execution.

| Model | Number of Runs | Mean Test Accuracy | Sample Std. Dev. | Minimum Test Accuracy | Maximum Test Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | Pending | Pending | Pending | Pending | Pending |
| Modified | Pending | Pending | Pending | Pending | Pending |

### Expected Checkpoints

- `checkpoints/pytorch/pytorch_baseline_seed_9486.pt`
- `checkpoints/pytorch/pytorch_baseline_seed_9487.pt`
- `checkpoints/pytorch/pytorch_baseline_seed_9488.pt`
- `checkpoints/pytorch/pytorch_modified_seed_9486.pt`
- `checkpoints/pytorch/pytorch_modified_seed_9487.pt`
- `checkpoints/pytorch/pytorch_modified_seed_9488.pt`

PyTorch per-run results, summary statistics, and curve diagnostics remain Pending.

# Neural Network Results

Status: Pending

## Baseline Results

Pending

## Modified Results

Pending

# CUDA Results

Status: Pending

## Compilation

Pending

## Timing

Pending

## Profiler

Pending
