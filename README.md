# Chicago Food Inspections Prediction

## Overview

This project utilizes data from the Chicago Department of Public Health to predict the outcome of food inspections. The aim is to assist inspectors in prioritizing establishments that are more likely to fail, thereby improving efficiency and ensuring public health compliance.

## Installation

To set up and run this project, you'll need to install several Python libraries which are essential for data handling, visualization, and modeling. If using Google Colab, you can install these directly:

```bash
pip install category_encoders numpy matplotlib pandas pdpbox scikit-learn xgboost
Project Structure
Import Data: Load the Chicago Food Inspections dataset.
Identify Data Leakage: Identify and remove any potential data leakage to ensure model validity.
Data Wrangling: Clean and preprocess data, handling missing values and removing irrelevant columns.
Split Data: Divide the dataset into training and validation sets.
Establish Baseline Accuracy: Determine baseline accuracy to gauge model improvements.
Model Building:
Bagging Model: Utilize RandomForestClassifier for predictions.
Boosting Model: Implement XGBClassifier for enhanced accuracy.
Model Evaluation:
ROC Curves: Optionally plot ROC curves to visualize model performance.
Classification Report: Summarize the accuracy, precision, and recall of the chosen model.
Permutation Importance: Analyze which features most significantly impact predictions.
PDP Interaction Plot (Optional): Examine how features like location interact to affect the model's predictions.
Key Results
Baseline Accuracy: Approximately 75%.
Bagging Model Performance:
Training Accuracy: 90.74%
Validation Accuracy: 68.13%
Boosting Model Performance:
Training Accuracy: 78.67%
Validation Accuracy: 70.13%
Important Features: Inspection Type, Latitude, and Longitude were found to be influential in predicting inspection outcomes.
Usage
To replicate the analysis or improve upon the models, follow these steps:

Clone the repository: git clone [repository-link]
Navigate to the project directory and install required packages.
Run the Jupyter Notebooks to perform the analysis from data wrangling to model evaluation.
Conclusion
The analysis demonstrates the potential of machine learning models to predict outcomes of food inspections, with the boosting model slightly outperforming the bagging model. The models identified key factors influencing inspection results, which can help inspectors focus on higher-risk establishments.

Next Steps
Hyperparameter Tuning: Refine models to improve accuracy and reduce overfitting.
Advanced Modeling Techniques: Explore more sophisticated algorithms or ensemble methods to enhance predictive performance.
Operational Integration: Implement the model into a real-time prediction tool for daily use by inspectors.
Contributing
Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
