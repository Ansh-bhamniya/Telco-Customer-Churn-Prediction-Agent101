# Telecom Customer Churn Prediction

You are the Senior Data Scientist at a telecom company. Your task is to build a robust customer churn prediction model using historical customer subscription and service usage data.

The dataset is provided in the directory `data/` and contains two CSV files:
- `customer_billing_churn.csv`
- `customer_profile.csv`

## Data Preparation

1. **Load `customer_billing_churn` and convert `TotalCharges` to a numeric data type, coercing errors to NaN. Fill any missing `TotalCharges` values with 0.**
2. In the `customer_billing_churn`, create spending and tenure–related features:
    - AvgMonthlySpend = TotalCharges / (tenure + EPS),
    - TenureChargeInteraction = MonthlyCharges * tenure,
    - *Note: `EPS` is a small constant (e.g., 1e-6) to avoid divide-by-zero.*
3. In the customer_billing_churn file, create a tenure stability indicator:
    - `ShortTenureFlag = 1` if `tenure < 12` else `0`.

## Dataset Integration

1. Combine all data in both input files on the basis of `customerID`.
2. Encode `Churn` as a binary variable (1 = "Yes", 0 = "No").
3. **One-hot encode all other categorical variables (e.g., `PaymentMethod`, `Contract`, `InternetService`, etc.) to convert them into numeric features.**
4. Name the combined and processed dataset as `churn_model_input`.

## Feature Selection and NA Handling

1. **Drop the `customerID` column from `churn_model_input` as it is not a predictive feature.**
2. Ensure there are no remaining missing values; drop rows if necessary.

## Model Setup 

1. Split the dataset into training and testing partitions using a **`train_test_split`** (70/30) with a random seed of 42 to ensure reproducibility.
2. Fit a Random Forest classification model.
3. Evaluate the model using F1 and AUC on the testing set.

## Deliverables
Return the following:
1. **`feature_importance_dict`** (Python dict) with feature → importance (rounded to 5 decimals).
2. **model_metrics** (Python dict) with keys: `f1_score` and `auc_score` (both rounded to 5 decimals).
3. **`churn_counts`** (pandas DataFrame) containing the counts of predicted classes (0 and 1) from the test set.

## Variable Serialization

Convert `churn_counts` DataFrame to a dictionary using `to_dict(orient='split')` before the end of your notebook.

