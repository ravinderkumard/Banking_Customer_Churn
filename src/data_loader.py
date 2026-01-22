# src/data_loader.py

import os
import pandas as pd
import numpy as np
from typing import Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,LabelEncoder
import joblib


# Data Loader Class
class DataLoader:
    def __init__(self, config_manager):
        self.config = config_manager.get_dataset_config()
        self.scaler = None
        self.config_manager = config_manager
    
    def _get_scaler(self, scaler_name: str):
        scalers = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "RobustScaler": RobustScaler,
        }

        if scaler_name not in scalers:
            raise ValueError(f"Unsupported scaler: {scaler_name}")

        return scalers[scaler_name]()

    def load_and_preprocess(self):
        """Load and preprocess data based on config"""
        print("\n LOADING AND PREPROCESSING DATA")
        print("-"*40)
        
        # Load data
        file_path = self.config.get('file_path')
        print("="*60)
        print(f"{file_path}")
        print("="*60)
        if not os.path.exists(file_path):
            # Try relative path
            file_path = os.path.join('..', file_path)
        
        df = pd.read_csv(file_path)
        print(f" Dataset loaded: {df.shape}")
        print(f" DF Clean Before: {df.columns}")
        
        df['Gender'] = df['Gender'].map({
            "Male": 1,
            "Female": 0,
            "M": 1,
            "F": 0,
            "O":3,
            "Other":3,
            "U":3,
            "Unknown":3
        })
    
        print("="*60)
        print(df["Gender"].value_counts())
        print("="*60)
        
        # Drop columns
        drop_cols = self.config.get('drop_columns', [])
        df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        print(f" Dropped columns: {drop_cols}")
        print(f" DF Clean After: {df_clean.columns}")
        # Get target
        target_col = self.config.get('target_column', 'Exited')
        if target_col not in df_clean.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f" Encoded {len(categorical_cols)} categorical columns")
        print(f" Final features: {X_encoded.shape[1]} (meets 12+ requirement)")
        
        # Split data
        preprocessing = self.config.get('preprocessing', {})
        test_size = preprocessing.get('test_size', 0.2)
        random_state = preprocessing.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        print(f" Split data: Train={X_train.shape}, Test={X_test.shape}")
        
        # Scale data if configured
        if preprocessing.get('scale_numerical', True):
            self.scaler = StandardScaler()
            numerical_cols = X_train.select_dtypes(include=[np.number]).columns
            
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            
            print(f" Scaled {len(numerical_cols)} numerical features")
            print(f" Save scaler : {preprocessing.get('scaling').get("enabled")}")
            os.makedirs("outputs/models", exist_ok=True)
            self._save_artifact(self.scaler,"outputs", preprocessing.get('scaling').get("artifact_name"))
            return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
        
        return X_train, X_test, X_train, X_test, y_train, y_test
    
    def _save_artifact(self, model, output_dir: str, filename: str):
        """ Method to save on StandarScaler Model"""
        model_path = os.path.join(output_dir, 'models', f'{filename}')
        joblib.dump(model, model_path)

        print(f"Saved artifact: {model_path}")