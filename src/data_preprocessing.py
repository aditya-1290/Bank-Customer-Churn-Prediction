import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the bank customer data"""
        return pd.read_csv(self.data_path)
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Check for missing values
        missing_values = df.isnull().sum()
        
        # For numerical columns, fill with median
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables using Label Encoding"""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        
        return df
    
    def scale_numerical_features(self, df):
        """Scale numerical features using StandardScaler"""
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced
    
    def prepare_data(self, test_size=0.2, handle_imbalance=True):
        """Full data preparation pipeline"""
        # Load data
        df = self.load_data()
        
        # Separate features and target
        y = df['Exited']
        X = df.drop('Exited', axis=1)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Encode categorical features
        X = self.encode_categorical_features(X)
        
        # Scale numerical features
        X = self.scale_numerical_features(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Handle class imbalance if requested
        if handle_imbalance:
            X_train, y_train = self.handle_class_imbalance(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
