from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from visualization import Visualizer

def main():
    # Initialize data preprocessor
    data_path = "data/Churn_Modelling.csv"
    preprocessor = DataPreprocessor(data_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    # Initialize and train models
    trainer = ModelTrainer()
    models = trainer.train_models(X_train, y_train)
    
    # Optimize the best performing model (example with random forest)
    best_params, best_score = trainer.optimize_model(
        X_train, y_train, 'random_forest'
    )
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    
    # Evaluate the optimized model
    best_model = trainer.models['random_forest']
    evaluation_results = trainer.evaluate_model(best_model, X_test, y_test)
    
    # Create visualizations
    visualizer = Visualizer()
    
    # Plot feature importance
    feature_names = X_train.columns
    visualizer.plot_feature_importance(best_model, feature_names)
    
    # Plot ROC curvea
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    visualizer.plot_roc_curve(y_test, y_pred_proba)
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(evaluation_results['confusion_matrix'])
    
    # Perform clustering on predicted churners
    predictions = best_model.predict(X_test)
    churner_segments, kmeans = trainer.cluster_churners(X_test, predictions)
    
    if churner_segments is not None:
        # Plot cluster analysis
        visualizer.plot_cluster_analysis(churner_segments, X_test.columns)

if __name__ == "__main__":
    main()
