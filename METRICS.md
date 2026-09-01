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

| Metric | Value |
| --- | ---: |
| Rows | 759 |
| Columns | 9 |
| Input features | 8 |
| Target column | 8 |
| Target classes | 0, 1 |
| Duplicate rows | 0 |
| Total explicit missing values | 0 |

## Target Class Distribution

| Class | Count | Percentage |
| ---: | ---: | ---: |
| 0 | 263 | 34.6509% |
| 1 | 496 | 65.3491% |

## Suspicious Zero Values

The supplied CSV is headerless, so the feature columns below use their standard order and numeric column indices. Zero is valid for pregnancy count (column 0) and the binary target (column 8). Suspicious measurement zeros are retained for inspection and will be handled during preprocessing.

| Column / standard feature | Zero count |
| --- | ---: |
| 0 / pregnancies | 111 |
| 1 / glucose | 5 |
| 2 / blood pressure | 35 |
| 3 / skin thickness | 224 |
| 4 / insulin | 371 |
| 5 / BMI | 11 |
| 6 / diabetes pedigree function | 1 |
| 7 / age-like feature | 63 |
| 8 / target | 263 |

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
