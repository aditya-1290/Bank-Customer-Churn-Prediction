# Bank Customer Churn Prediction

## Project Overview
This project implements a machine learning solution to predict customer churn in a banking context. It uses various ML algorithms to identify customers likely to leave the bank and segments them to understand different patterns of churning behavior.

## Features
- Data preprocessing and feature engineering
- Handling of imbalanced datasets using SMOTE
- Multiple ML models implementation:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
- Advanced customer segmentation using K-means clustering
- Comprehensive visualization suite
- Model performance evaluation using various metrics

## Project Structure
```
project_bank/
├── data/
│   └── Churn_Modelling.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── visualization.py
│   └── main.py
└── requirements.txt
```

## Installation

1. Create a virtual environment (optional but recommended):
```bash
python -m venv bank_env
source bank_env/bin/activate  # On Windows: bank_env\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset (`Churn_Modelling.csv`) in the `data` directory.
2. Run the main script:
```bash
cd src
python main.py
```

## Dataset Requirements
The `Churn_Modelling.csv` should contain the following information:
- Customer demographic data
- Account information
- Transaction history
- Target variable 'Exited' (1 for churned customers, 0 for retained)

## Model Components

### Data Preprocessing (`data_preprocessing.py`)
- Handles missing values
- Encodes categorical variables
- Scales numerical features
- Implements SMOTE for handling class imbalance

### Model Training (`model_training.py`)
- Implements multiple classification algorithms
- Performs hyperparameter optimization
- Includes model evaluation metrics
- Implements customer segmentation

### Visualization (`visualization.py`)
- Feature importance plots
- ROC curves
- Confusion matrices
- Cluster analysis visualizations

### Main Script (`main.py`)
Orchestrates the entire workflow:
1. Data loading and preprocessing
2. Model training and optimization
3. Performance evaluation
4. Customer segmentation
5. Visualization generation

## Model Evaluation
The project evaluates models using:
- ROC-AUC score
- Precision-Recall curves
- Confusion matrices
- Cross-validation scores

## Customer Segmentation
After predicting potential churners, the project performs K-means clustering to identify distinct groups of customers who might churn, helping in:
- Understanding different patterns of churning behavior
- Developing targeted retention strategies
- Identifying high-value customers at risk

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn

## Future Improvements
- Implementation of more advanced algorithms
- Feature selection optimization
- API development for real-time predictions
- Integration with a dashboard for visualization
- Deployment pipeline setup

## Contributing
Feel free to contribute to this project:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
