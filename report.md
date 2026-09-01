# DATA 266 — HW1

| Parameter | Value |
| --------- | ----: |
| SID4      |  9486 |
| SEED      |  9486 |
| SLICE     |   486 |
| HP\_ID    |     0 |
| CLS\_A    |     6 |
| CLS\_B    |     0 |

## 1. Autoregressive Models

An autoregressive (AR) model is a statistical model that uses the previous values of a time series to predict its current or future value. A time series is a collection of observations recorded sequentially over time, such as daily temperatures, monthly sales, electricity demand, or stock prices.

The previous observations used by the model are called lagged values. For example, a lag of 1 refers to the immediately preceding observation, while a lag of 2 refers to the observation two time steps earlier. An autoregressive model of order \(p\), written as AR(\(p\)), uses the previous \(p\) values:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \epsilon_t
$$

Here, \(y_t\) is the value being predicted, \(c\) is a constant, and the \(\phi\) terms are learned coefficients that determine how strongly each lagged value influences the prediction. The term \(\epsilon_t\) represents unpredictable error, and \(p\) is the number of lagged values used.

For example, an AR(2) temperature model predicts today's temperature using the temperatures from yesterday and two days ago. AR models can also forecast electricity demand from previous demand, forecast future sales from earlier sales, analyze financial values using their recent history, and predict weather measurements from previous measurements.

AR models are relatively simple and computationally efficient, and their coefficients are interpretable. They often capture useful short-term dependencies in sequential data. They work best when the time series is reasonably stationary, meaning properties such as its mean and variance remain stable over time.

Source: DATA 266 course lecture, “01-Intro-NN-CUDA,” slides 54–59.

## 2. Diabetes Dataset and Neural Networks

### 2.1 Dataset Overview

The supplied diabetes CSV contains 759 rows and 9 source columns. Because it has no header row, semantic names are assigned using the matched dataset schema: the first 8 columns are input features and the final column is `SourceLabel`. The source label contains 263 zeros and 496 ones; its encoding is `0 = diabetes` and `1 = no diabetes`. For modeling, a new `Diabetes` target is created as `1 - SourceLabel`, so `Diabetes=1` means diabetes. The remapped target contains 496 class-0 observations (65.3491%) and 263 class-1 observations (34.6509%).

The dataset contains no explicit missing values and no duplicate rows. Its features arrived already scaled approximately to `[-1, 1]`. Zero is valid for Pregnancies, DiabetesPedigreeFunction, and Age, while zero is a missing-value sentinel for Glucose (5), BloodPressure (35), SkinThickness (224), Insulin (371), and BMI (11). SourceLabel zeros are also valid source labels. These five sentinel columns are converted to missing values for preprocessing; valid zeros are preserved.

### 2.2 Preprocessing and Data Split

The provided features were already scaled approximately to `[-1, 1]`, but the five medically invalid zero sentinels are first replaced with missing values. The source label is remapped with `Diabetes = 1 - SourceLabel`, making class 1 the diabetes class. Pregnancies, DiabetesPedigreeFunction, and Age retain their valid zeros.

Using `random_state=9486` and stratification, row indices are split once into 531 training rows, 114 validation rows, and 114 testing rows. A median imputer is fit only on the training features, then used to transform validation and test features. A `StandardScaler` is likewise fit only on the imputed training features and then applied to all three subsets. This creates leakage-free float32 arrays that will be shared by the PyTorch and TensorFlow experiments. The exact split sizes and class counts are recorded in `METRICS.md`; model results are not yet available.

### 2.3 Exploratory Data Analysis

For exploratory analysis, the full dataset is used descriptively rather than for model fitting. Zero is replaced by missing values only in Glucose, BloodPressure, SkinThickness, Insulin, and BMI; valid zeros in the other features are preserved. The Pearson correlations with `Diabetes`, sorted by absolute magnitude, are Glucose (0.491538), BMI (0.315051), Insulin (0.303797), SkinThickness (0.262079), Pregnancies (0.218405), BloodPressure (0.169221), DiabetesPedigreeFunction (0.163246), and Age (0.112565). These are positive, mostly weak-to-moderate linear associations and do not establish causation.

The class distributions overlap across all eight feature plots. The Diabetes=1 group appears shifted toward higher scaled values for Glucose, BMI, Insulin, and Pregnancies, but these are descriptive visual patterns rather than claims of statistical significance. Insulin and SkinThickness have substantial missing-sentinel counts, so their distributions use fewer observations and require caution. The target classes are moderately imbalanced: Diabetes=0 has 496 observations and Diabetes=1 has 263.

![Correlation matrix](figures/correlation_matrix.png)

![Feature distributions](figures/feature_distributions.png)

### 2.4 PyTorch Experimental Method

The PyTorch baseline uses hidden layers `[64, 32]`, while the HP_ID 0 modified model reduces capacity to `[32]`. Both models use ReLU hidden activations, one raw output logit, Adam with learning rate `0.001`, `BCEWithLogitsLoss`, batch size `32`, and exactly 30 epochs. The fixed preprocessed training, validation, and testing split from Step 3 is reused without re-splitting, and the three training seeds are `9486`, `9487`, and `9488`.

Each run records sample-weighted training and validation losses and evaluates test accuracy only after training using a sigmoid threshold of `0.5`. The mean, minimum, maximum, and sample standard deviation of the three test accuracies will be reported after execution, with the standard deviation calculated using `np.std(accuracies, ddof=1)`. Each model/seed run will save a checkpoint containing the model state and experiment metadata. The per-run results, summary statistics, and loss-curve interpretation remain pending Colab execution.
