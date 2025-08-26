from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42)
        }
        self.best_model = None
        self.best_score = 0
        
    def train_models(self, X_train, y_train):
        """Train all models and keep track of the best one"""
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                
        return self.models
    
    def optimize_model(self, X_train, y_train, model_name):
        """Perform hyperparameter optimization using GridSearchCV"""
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance using various metrics"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC-AUC score
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix
        }
    
    def cluster_churners(self, X, predictions, n_clusters=3):
        """
        Perform customer segmentation on predicted churners using K-means clustering
        """
        # Filter data for predicted churners
        churner_data = X[predictions == 1]
        
        if len(churner_data) == 0:
            return None, None
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(churner_data)
        
        # Add cluster labels to the churner data
        churner_segments = pd.DataFrame(churner_data)
        churner_segments['Cluster'] = cluster_labels
        
        return churner_segments, kmeans
