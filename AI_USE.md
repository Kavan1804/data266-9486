# Question 1
An AI assistant helped initialize the repository and notebook structure for Step 0, organize the autoregressive-model explanation using the provided lecture material for Step 1, generate the dataset-inspection code, organize the notebook cells, and summarize the initial dataset observations for Step 2. For Step 3, an AI assistant helped recognize the scaled dataset format, clarify the label direction, create leakage-free preprocessing and split code, and organize verification assertions. For Step 4, an AI assistant helped create visualization code, improve plot layout, and organize the correlation and distribution interpretations. For Step 5, an AI assistant helped structure the PyTorch model, reusable training loop, reproducibility controls, metrics table, and checkpoint logic. For the executed-results integration, an AI assistant helped verify and organize the Colab outputs and artifacts. For Step 6, an AI assistant helped structure the TensorFlow/Keras model, raw-logit loss configuration, deterministic seeding, repeated-run loop, summary calculations, and checkpoint logic.

# Question 2
The assistant helped organize the implementation and documentation, but the reported training values came from the executed Colab notebook and the supplied artifact archive.

# Question 3
The assistant made this specific mistake earlier: “Age-like feature, column 7: 63 suspicious zeros.”

The assistant initially treated scaled age zeros as missing. Schema verification showed that zero was a valid scaled age value, not automatically missing. The issue was found by examining value ranges, matching rows with the scaled dataset schema, and checking which raw medical zeros were genuinely invalid. The correction preserved zeros in Pregnancies, DiabetesPedigreeFunction, and Age, while treating zero as missing only in Glucose, BloodPressure, SkinThickness, Insulin, and BMI. This prevents valid observations from being incorrectly imputed.

# Question 4
The work was verified by comparing the notebook outputs with the supplied values, validating the artifact filenames and sizes, inspecting the PNG dimensions and readability, and cross-checking the documented metrics against the executed notebook outputs. The ZIP itself was not committed.
