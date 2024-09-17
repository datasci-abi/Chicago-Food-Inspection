Chicago Food Inspections Prediction
Overview
This project focuses on using machine learning techniques to predict the outcomes of food safety inspections in Chicago. The aim is to build a model that predicts whether an establishment will fail an inspection, thereby helping inspectors prioritize their time efficiently. The dataset comes from the Chicago Department of Public Health.

The project follows tasks that guide the analysis and modeling process, covering data wrangling, feature engineering, model training, and evaluation.

Installation
If you're working in Google Colab, run the following commands to install the necessary packages:

python
Copy code
%%capture
import sys

if 'google.colab' in sys.modules:
    !pip install category_encoders
    !pip install matplotlib==3.7.1
    !pip install pdpbox
    !pip install scikit-learn==1.1.3
    !pip install --upgrade pdpbox
You will need the following libraries to run the code in this project:

category_encoders
numpy
matplotlib
pandas
pdpbox
sklearn
xgboost
Tasks Overview
This project includes 12 tasks that cover various stages of the data science process:

Import Data: Load and examine the dataset.
Identify Data Leakage: Find and remove features that would cause data leakage.
Data Wrangling: Prepare the dataset for modeling by handling missing values and dropping unneeded columns.
Feature & Target Split: Split the data into the feature matrix X and target vector y.
Training & Validation Split: Divide the dataset into training and validation sets.
Baseline Accuracy: Calculate the baseline accuracy to compare model performance.
Bagging Predictor: Build a bagging model using RandomForestClassifier.
Boosting Predictor: Build a boosting model using XGBClassifier from XGBoost.
ROC Curve: (Stretch Goal) Plot the ROC curve for both models.
Classification Report: Generate a classification report for the best model.
Permutation Importance: Calculate permutation importances to understand feature impact.
PDP Interaction Plot: (Stretch Goal) Create a Partial Dependence Plot (PDP) to examine feature interactions.
How to Run
Wrangle Data: The data is imported and cleaned, including handling of missing values, dropping high-cardinality columns, and identifying leaky features. The wrangling function is designed to clean and preprocess the dataset for machine learning.

Example:

python
Copy code
def wrangle(df):
    df = df.drop(columns='Serious Violations Found')
    high_cardinality_cols = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() > 500]
    df = df.drop(columns=high_cardinality_cols)
    return df
Feature Engineering: Split the data into features X and target y.

Example:

python
Copy code
X = df.drop(columns='Fail')
y = df['Fail']
Model Training: You will train two models—a bagging predictor using RandomForestClassifier and a boosting predictor using XGBClassifier. A Pipeline is used to handle preprocessing and model fitting.

Example for RandomForest model:

python
Copy code
model_bag = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
model_bag.fit(X_train, y_train)
Evaluation: After training the models, evaluate them using accuracy scores and ROC curves, then choose the best model for further analysis.

Example for generating ROC curve:

python
Copy code
plt.plot(fpr_bag, tpr_bag, label='Bagging Model')
plt.plot(fpr_boost, tpr_boost, label='Boosting Model')
Results
Baseline Accuracy: Established baseline accuracy score is approximately 75%.
Bagging Model Accuracy: Achieved a training accuracy of 90.7% and a validation accuracy of 68.1%.
Boosting Model Accuracy: Achieved a training accuracy of 78.7% and a validation accuracy of 70.1%.
Feature Importances: Used permutation importance to identify key features like Inspection Type, Latitude, and Longitude.
Partial Dependence Plot: Visualized feature interactions between Latitude and Longitude to assess geographical risk factors.
Conclusion
In this project, we built and evaluated two models to predict the outcome of food safety inspections. The boosting model performed slightly better than the bagging model on the validation set. Insights gained from feature importance and partial dependence plots suggest geographic location and inspection type play a crucial role in inspection outcomes.

Next Steps
Explore additional data preprocessing techniques for better model performance.
Experiment with hyperparameter tuning to improve model accuracy.
Apply more advanced ensemble techniques like stacking to further enhance predictions.
Feel free to clone the repository and run the code on your local environment or Google Colab.
