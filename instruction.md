You are given a dataset for predicting customer churn in a telecom company (binary classification task).  
The training data 'train.csv' is in the directory 'data/' with the following columns:  
customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn  

'Churn' is our target column (Yes/No) — the rest are the available features.  
Complete the class `ChurnPredictor` in the given initial notebook.  
You have 'torch', 'scikit-learn', 'xgboost' and other common libraries available.  
Note that both 'fit' and 'predict' functions should take dataframes as input (and series for target during fit) as shown in the given initial notebook.

**Data Split**:  
Split the 'train.csv' data into 80% for training and 20% for validation.  
Train your model on the 80% split and evaluate on the 20% validation split.

**Binary Classification Note**:  
The 'Churn' column is already binary:  
- "Yes" → customer will churn (label as 1)  
- "No" → customer will stay (label as 0)  
No need to create a new target column — use 'Churn' directly (map "Yes"→1, "No"→0 if needed).

**Goal**:  
Achieve ≥ 0.83 ROC AUC score on the 20% validation split (measured using 'predict_proba' on the positive class).

**Required Methods**:
1. 'fit(train_df)'  
   - Takes a full dataframe (including the 'Churn' column) and trains the model.  
   - You may perform any preprocessing, encoding, scaling, imputation, etc., inside this method.

2. 'predict(df)'  
   - Takes a dataframe (without the 'Churn' column)  
   - Returns binary predictions (0 or 1) as a NumPy array or list of length n_samples.

3. 'predict_proba(df)'  
   - Takes a dataframe (without the 'Churn' column)  
   - Returns predicted probabilities as a NumPy array of shape (n_samples, 2),  
     where column 1 contains probabilities for class 1 (churn).

**Note**:  
The model is evaluated using ROC AUC score calculated independently by the test suite using 'predict_proba()' output (column 1) and true labels from the held-out test set.




**Critical Final Step — MUST DO**:
After completing and testing your model in the notebook:
- Run ALL cells to verify everything works.
- **MANDATORY**: Execute the last cell to create the '/results' directory (if it doesn't exist) and **write the entire ChurnPredictor class (with all imports and helper functions)** to the file '/results/utils.py'.
- This file **MUST** be created — the verifier will load it directly and test it.
- Do NOT skip this step — if /results/utils.py is missing, all tests will fail.
- Confirm by printing: "Successfully wrote ChurnPredictor to /results/utils.py"




The model will be directly loaded from the python file and used as below (rough code):

```python
from utils import ChurnPredictor

model = ChurnPredictor()
model.fit(train_df)
y_pred = model.predict(df_features)     # df_features is dataframe without 'Churn' column
y_pred_proba = model.predict_proba(df_features)
...