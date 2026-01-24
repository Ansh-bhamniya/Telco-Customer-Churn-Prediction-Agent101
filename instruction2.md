# Task Instructions

You are the Senior Data Scientist at a manufacturing company. Your task is to build a daily raw-material cost prediction model using the following
three datasets:

- `df_rm` - Raw material consumption
- `df_fx` - Daily FX rates
- `df_sales` - Sales data

## Data Preparation

1. Keep only `raw_material_code == "IRON_ORE"` in raw material data and `confirmed` status in sales data.
2. Convert all three datasets into daily-level data ensuring that days with multiple entries are aggregated appropriately (e.g., summing quantities, averaging `cost_per_kg`). Let daily sales in kg be `daily_sales_kg` for sales data and daily consumption in kg be `kg_consumed` in raw material data.
3. In the FX dataset, if multiple entries appear for the same day and currency, retain only the last valid entry for that day.

## Dataset Integration

1. Combine all data on date and currency level.
2. Compute target variable: `rm_cost_usd = cost_per_kg_local * kg_consumed * rate_to_USD`.
3. Create lag features: `lag_kg_consumed_1d` and `lag_daily_sales_kg_1d`.
4. Feature selection and NA handling: Use exactly six features: `cost_per_kg_local`, `kg_consumed`, `rate_to_USD`, `daily_sales_kg`, `lag_kg_consumed_1d`, `lag_daily_sales_kg_1d`. Drop any row containing NA in features or target.

## Model Setup

1. Split the dataset into training and testing partitions using a `train_test_split` (70/30) with a random seed of 42 to ensure reproducibility.
2. Fit a standard linear regression model (no regularization).
3. Evaluate the model using RMSE and R² on the testing set.

## Deliverables

Return the following:

1. **`coefficients_dict`** (Python dict) with keys: `intercept`, `cost_per_kg_local`, `kg_consumed`, `rate_to_USD`, `daily_sales_kg`, `lag_kg_consumed_1d`, `lag_daily_sales_kg_1d` (all rounded to 4 decimals).

2. **`model_quality`** (Python dict) with keys: `rmse` and `r2` (both rounded to 4 decimals).

3. **`daily_prod`** (pandas DataFrame) containing daily aggregated sales data with columns:
    - `date`: Date column
    - `daily_sales_kg`: Daily sales in kilograms

## Variable Serialization

Convert `daily_prod` DataFrame to a dictionary using `to_dict(orient='split')` before the end of your notebook.

## Data Cleaning Notes

The `quantity_kg` column in the sales data contains non-numeric string values that must be parsed into numbers.