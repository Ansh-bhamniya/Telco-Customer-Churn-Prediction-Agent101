
You are solving a customer churn prediction task for a telecom company.



**Task:**
- Training data: `/workspace/data/train.csv`
- Goal: Train a model to predict which customers will churn
- Target: Achieve ROC AUC score >= 0.83

**Simple Steps:**
1. Load training data from `/workspace/data/train.csv`
2. Handle data issues (convert TotalCharges to numeric, handle missing values)
3. Split data: 80% for training, 20% for validation
4. Preprocess: Encode categorical variables, scale numeric features
5. Train a model (RandomForestClassifier or XGBoost)
6. Evaluate on validation set (ROC AUC should be >= 0.83)

**Required Output:**
Create `/results/utils.py` with:
- All necessary imports
- A class with `fit()`, `predict()`, and `predict_proba()` methods

**Output Format:**
When you print results, show only two columns:
- Column 1: Sample ID or index
- Column 2: Prediction result (0 or 1 for churn)

Example output:
```
ID    Prediction
0     0
1     1
2     0
```

**Constraints:**
- Use only available libraries (pandas, scikit-learn, numpy, etc.)
- No new installations
- Export your class to `/results/utils.py`
