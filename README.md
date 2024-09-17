
Chicago Food Inspections Prediction
Overview
This project focuses on predicting whether a food establishment in Chicago will pass or fail a health inspection. The goal is to help inspectors use their time more efficiently by predicting which places are most likely to fail. The dataset comes from the Chicago Department of Public Health.

Installation
To run this project, you need to install the following libraries:

category_encoders
numpy
matplotlib
pandas
pdpbox
scikit-learn
xgboost
If you are working in Google Colab, install these libraries using pip commands.

Project Structure
This project is divided into several tasks that cover data analysis, modeling, and evaluation:

Import Data: Load and examine the dataset to understand its structure.
Identify Data Leakage: Identify and remove columns that could cause data leakage, such as information that wouldn’t be available before an inspection.
Data Wrangling: Clean and prepare the dataset by handling missing values and removing unnecessary columns.
Split Data: Separate the dataset into features (input) and target (output).
Train-Test Split: Divide the data into training and validation sets to train and test models.
Baseline Accuracy: Establish a baseline accuracy to evaluate the performance of models.
Bagging Model: Train a model using the bagging technique (RandomForestClassifier).
Boosting Model: Train a model using the boosting technique (XGBoost).
ROC Curve: (Optional) Plot the ROC curves for both models to visualize performance.
Classification Report: Generate a classification report for the model that performs better.
Permutation Importance: Analyze which features have the most impact on model predictions.
PDP Interaction Plot: (Optional) Create a Partial Dependence Plot (PDP) to see how specific features like location affect predictions.
Results
Baseline Accuracy: The model’s baseline accuracy is approximately 75%.
Bagging Model: Achieved a validation accuracy of 68.1%.
Boosting Model: Performed better, with a validation accuracy of 70.1%.
Feature Importance: Features such as Inspection Type, Latitude, and Longitude were found to be key factors in predicting the outcome of inspections.
Conclusion
In this project, we built and evaluated two models—bagging and boosting—to predict food safety inspection results. The boosting model performed slightly better, and feature analysis indicated that geographic location and the type of inspection play a significant role in determining whether a place will pass or fail.

Next Steps
Experiment with hyperparameter tuning to improve model performance.
Explore more advanced machine learning techniques for better predictions.
Use the findings to help inspectors prioritize inspections and improve public health outcomes.
Feel free to explore the project and try improving the models by cloning the repository!
