import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


class ChurnPredictor:
    '''
    Simple churn prediction model.
    Trains on training data and provides predictions.
    '''
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pipeline = None
        self.feature_columns = None
    
    def fit(self, train_df):
        '''Train the model. Internally splits data 80/20 for training.'''
        # Make a copy
        df = train_df.copy()
        
        # Handle TotalCharges - convert to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Drop customerID (not a feature)
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'])
        
        # Internal 80/20 split as required
        train_split, val_split = train_test_split(
            df, 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=df['Churn']
        )
        
        # Prepare features and target for training
        X_train = train_split.drop(columns=['Churn'])
        y_train = (train_split['Churn'] == 'Yes').astype(int)
        
        # Store feature columns for later
        self.feature_columns = X_train.columns.tolist()
        
        # Identify column types
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Build preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        # Build full pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('model', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Optional: Check validation performance
        X_val = val_split.drop(columns=['Churn'])
        y_val = (val_split['Churn'] == 'Yes').astype(int)
        y_val_proba = self.pipeline.predict_proba(X_val)[:, 1]
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        print(f'Validation ROC AUC: {val_roc_auc:.4f}')
        
        return self
    
    def predict(self, X):
        '''Predict churn class (0 or 1).'''
        if self.pipeline is None:
            raise RuntimeError('Call fit() before predict()')
        
        X_df = X.copy()
        
        # Drop customerID if present
        if 'customerID' in X_df.columns:
            X_df = X_df.drop(columns=['customerID'])
        
        # Handle TotalCharges
        if 'TotalCharges' in X_df.columns:
            X_df['TotalCharges'] = pd.to_numeric(X_df['TotalCharges'], errors='coerce')
        
        # Align columns with training
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in X_df.columns:
                    X_df[col] = np.nan
            X_df = X_df[self.feature_columns]
        
        return self.pipeline.predict(X_df)
    
    def predict_proba(self, X):
        '''Predict churn probabilities. Returns (n_samples, 2) array.'''
        if self.pipeline is None:
            raise RuntimeError('Call fit() before predict_proba()')
        
        X_df = X.copy()
        
        # Drop customerID if present
        if 'customerID' in X_df.columns:
            X_df = X_df.drop(columns=['customerID'])
        
        # Handle TotalCharges
        if 'TotalCharges' in X_df.columns:
            X_df['TotalCharges'] = pd.to_numeric(X_df['TotalCharges'], errors='coerce')
        
        # Align columns with training
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in X_df.columns:
                    X_df[col] = np.nan
            X_df = X_df[self.feature_columns]
        
        return self.pipeline.predict_proba(X_df)
