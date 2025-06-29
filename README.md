# Credit Default Prediction
An end-to-end machine learning pipeline to predict customer loan default, with model explainability and Power BI dashboard integration.

## 📦 Project Structure

- `PreProcessing_PD_Model.py`: Python module with all preprocessing steps (outlier removal, encoding, scaling, SMOTE)
- `Default_prediction_model.ipynb`: Jupyter notebook for training, testing, and exporting ML models
- `logisticPDmodel.pkl`,`rfPDmodel.pkl` & `XGBPDmodel.pkl`: Trained models for deployment
- `pd_prediction.xlsx`: Input prediction data (with same schema as training data)
- `ClassificationModelMonitoring.pbix`: Power BI dashboard for live model monitoring

## 🔁 Preprocessing Pipeline

- Removes outliers (`person_age < 70`, `emp_length < 52`)
- Fills missing values in `loan_int_rate` with mean
- One-hot encodes categorical variables (`loan_intent`, `loan_grade`, `home_ownership`)
- Maps binary default field
- Scales continuous features using `StandardScaler`
- Balances dataset using `SMOTE`

## 🧪 Model Training

- Logistic Regression - Interpretable baseline model
- Random Forest Classifier - Handles nonlinearities and ranks feature importance
- XGBoost Performance-optimized gradient boosting model
- Evaluation via confusion matrix and classification report

## 📊 Power BI Dashboard

- Visualizes prediction results and feature distributions
- Monitors model outputs over time

## 🛠 Tech Stack

- Python (pandas, numpy, sklearn, imblearn, xgboost)
- Power BI
- Jupyter Notebook
- Pickle (model persistence)

