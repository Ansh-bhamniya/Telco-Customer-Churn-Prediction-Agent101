#!/usr/bin/env python3
"""
Split the Telco Customer Churn dataset:
1. Split WA_Fn-UseC_-Telco-Customer-Churn.csv into train.csv (80%) and test.csv (20%)
2. The train.csv will be used for training, and should be split into 80% training / 20% validation internally
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Read the original dataset
print("Reading original dataset...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"Total rows: {len(df)}")

# First split: train.csv (80%) and test.csv (20%) - test.csv is the final holdout
print("\nSplitting into train.csv (80%) and test.csv (20%)...")
train_size = 0.8
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(shuffled_df) * train_size)

df_train = shuffled_df[:split_idx].copy()
df_test = shuffled_df[split_idx:].copy()

print(f"Train set: {len(df_train)} rows")
print(f"Test set: {len(df_test)} rows")

# Create directories if they don't exist
Path('data').mkdir(exist_ok=True)
Path('test').mkdir(exist_ok=True)
Path('tests').mkdir(exist_ok=True)  # Some tests might expect /tests/

# Save train.csv to data/ directory
train_path = Path('data/train.csv')
df_train.to_csv(train_path, index=False)
print(f"\nSaved train.csv to {train_path}")

# Save test.csv to test/ directory (and tests/ for compatibility)
test_path = Path('test/test.csv')
tests_path = Path('tests/test.csv')
df_test.to_csv(test_path, index=False)
df_test.to_csv(tests_path, index=False)
print(f"Saved test.csv to {test_path}")
print(f"Saved test.csv to {tests_path} (for compatibility)")

# Verify the splits
print("\nVerification:")
print(f"Train set Churn distribution:")
print(df_train['Churn'].value_counts())
print(f"\nTest set Churn distribution:")
print(df_test['Churn'].value_counts())

print("\n✓ Data split completed successfully!")
print("\nNote: The train.csv should be split into 80% training / 20% validation")
print("      inside the ChurnPredictor.fit() method for model development.")
