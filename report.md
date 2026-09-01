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

The supplied diabetes CSV contains 759 rows and 9 columns. Because it has no header row, the columns are indexed from 0 through 8. The first 8 columns are input features, and column 8 is the final binary target column representing the diabetes outcome. The target contains 263 class-0 observations (34.6509%) and 496 class-1 observations (65.3491%).

The dataset contains no explicit missing values and no duplicate rows. It does contain zero values that may be suspicious for medical measurements: under the standard feature order, glucose (column 1) has 5, blood pressure (column 2) has 35, skin thickness (column 3) has 224, insulin (column 4) has 371, and BMI (column 5) has 11. The age-like column 7 also contains 63 zeros. Zero is valid for the pregnancy-count feature and for the binary target. These suspicious zeros will be handled during preprocessing rather than changed during inspection. Preprocessing and data splitting will be performed in the next stage; no model accuracy or preprocessing results are reported here.
