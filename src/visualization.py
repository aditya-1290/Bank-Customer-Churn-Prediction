import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    @staticmethod
    def plot_feature_importance(model, feature_names):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'features': feature_names,
                'importance': model.feature_importances_
            })
            importances = importances.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importances.head(10), x='importance', y='features')
            plt.title('Top 10 Most Important Features')
            plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(conf_matrix):
        """Plot confusion matrix as a heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    @staticmethod
    def plot_cluster_analysis(churner_segments, feature_cols):
        """Plot cluster analysis for churned customers"""
        # Calculate cluster means
        cluster_means = churner_segments.groupby('Cluster')[feature_cols].mean()
        
        # Create heatmap of cluster characteristics
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_means, annot=True, cmap='coolwarm', center=0)
        plt.title('Cluster Characteristics Heatmap')
        plt.show()
        
        # Create boxplots for key features across clusters
        for feature in feature_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=churner_segments, x='Cluster', y=feature)
            plt.title(f'{feature} Distribution Across Clusters')
            plt.show()
