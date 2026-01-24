# Telecom Customer Churn Prediction

You are given a dataset of customer subscription and service usage records for a telecom company in the directory `data/`. 
The directory consists of:

* **train.csv**: Contains columns `['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']`.
* **test.csv**: Contains only the features `['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']`. **The target 'Churn' column has been removed for evaluation.**

**Goal**: Achieve a ROC-AUC score greater than 0.83 on the test dataset.

### Deliverables

Complete the class `ChurnPredictor` in the initial notebook. Once the model is complete, **write the entire class** along with necessary imports into the file `/results/utils.py`. Make sure to create the directory `/results` if it does not exist.

* Your class must handle `fit(X, y)` and `predict(X)` taking DataFrames as input.
* **Robustness**: Your model must automatically ignore any extra or "garbage" columns present in the input DataFrame during prediction.
* The model will be instantiated and tested automatically:

    ```python
    from utils import ChurnPredictor

    model = ChurnPredictor()  # Ensure default arguments work
    model.fit(X_train, y_train)
    ```
