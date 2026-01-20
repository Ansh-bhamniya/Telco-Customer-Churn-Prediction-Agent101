You are an autonomous AI data scientist solving a customer churn prediction task for a telecom company.

The training data `'train.csv'` is located in `/workspace/data/train.csv` with the following columns:
`customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn`

The situation:
we have customers in the directory (`/workspace/data/train.csv`) we want to analyze which customers are likely to cancel their plans so the company can take prevention action. The model will be tested on a hidden test.csv to measure real performance using ROC AUC score. The target performance is ROC AUC >= 0.83.

Your Objective:
1. Analyze the data and implement a complete ChurnPredictor class that trains on the dataset and produces correct predictions and probabilities.
2. The class must handle common data issues (missing values, non-numeric TotalCharges, categorical encoding) and follow the exact interface.
3. The model must achieve ROC AUC score >= 0.83 on the validation set (20% split) to meet the performance requirement.

Constraints:
1. Use only available libraries (Pandas, scikit-learn, xgboost, numpy, matplotlib, etc.)
2. No new installations
3. Class must be self-contained and follow the exact interface.

Required Outputs:
1 file: `/results/utils.py` containing class 'ChurnPredictor' with exactly these methods:
- fit(self, train_df: pd.DataFrame): Train on full dataframe (including 'Churn'). Internally split into 80% training and 20% validation. Train on 80% and evaluate on 20% validation split.
- predict(self, X: pd.DataFrame): Return binary prediction (0/1) as NumPy array or list
- predict_proba(self, X: pd.DataFrame): Return probabilities as NumPy array of shape (n_samples, 2), where column index 1 contains probabilities for class 1 (churn)
