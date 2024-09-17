Chicago Food Inspections Prediction
Overview
This project aims to predict whether a food establishment in Chicago will pass or fail a health inspection. The goal is to help inspectors prioritize their time by predicting which places are most likely to fail. The dataset is from the Chicago Department of Public Health.

Installation
To run this project, you need to install the following libraries:

category_encoders
numpy
matplotlib
pandas
pdpbox
scikit-learn
xgboost
If using Google Colab, install these libraries with pip.

Project Structure
The project is divided into several tasks covering data analysis, modeling, and evaluation:

Import Data: Load and examine the dataset.
Identify Data Leakage: Remove columns that could cause data leakage (information not available before the inspection).
Data Wrangling: Clean the dataset by handling missing values and removing unnecessary columns.
Split Data: Divide the data into features (input) and target (output).
Train-Test Split: Split the data into training and validation sets.
Baseline Accuracy: Establish a baseline accuracy to compare model performance.
Bagging Model: Train a model using RandomForestClassifier.
Boosting Model: Train a model using XGBoost.
ROC Curve (Optional): Plot ROC curves for both models.
Classification Report: Generate a report for the best model.
Permutation Importance: Identify the most important features for prediction.
PDP Interaction Plot (Optional): Create a Partial Dependence Plot to see how location affects predictions.
Results
Baseline Accuracy: Around 75%.
Bagging Model: Validation accuracy of 68.1%.
Boosting Model: Validation accuracy of 70.1%.
Key Features: Inspection Type, Latitude, and Longitude were important in predicting the inspection outcome.
Conclusion
In this project, two models (bagging and boosting) were built to predict food inspection results. The boosting model performed slightly better, and feature analysis showed that geographic location and inspection type were key factors in determining whether an establishment passes or fails.

Next Steps
Experiment with hyperparameter tuning to improve model performance.
Explore more advanced machine learning techniques for better predictions.
Use these insights to help inspectors focus on higher-risk establishments.
Feel free to explore the project and try improving the models by cloning the repository!
