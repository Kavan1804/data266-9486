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
