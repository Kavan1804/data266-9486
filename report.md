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
