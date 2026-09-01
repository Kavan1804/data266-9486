# Question 1

I used an AI assistant mainly to validate my work, check consistency between notebook outputs and reported metrics, and improve the organization of Markdown documentation. I wrote and executed the notebook experiments in Colab, reviewed the results, and made the final decisions about preprocessing, interpretation, and conclusions.

# Question 2

One incorrect assistant-generated observation was: “Age-like feature, column 7: 63 suspicious zeros.” This incorrectly assumed that zeros in the scaled Age feature represented missing data.

# Question 3

I discovered the mistake by examining the feature ranges and comparing the dataset structure with the scaled diabetes schema. This showed that zero was a valid scaled Age value, while zeros were invalid medical measurements only for Glucose, BloodPressure, SkinThickness, Insulin, and BMI.

# Question 4

I corrected the preprocessing to preserve zeros in Age, Pregnancies, and DiabetesPedigreeFunction. Only invalid zeros in the five medical-measurement columns were converted to missing values and imputed using training-set medians, preventing valid observations from being altered.