"""
Configuration for Smart Predictor pipelines
"""

# Diabetes dataset configuration
DIABETES_CONFIG = {
    "target_column": "Diabetes_binary",
    "problem_type": "binary_classification",
    "feature_columns": [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
        "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", 
        "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
        "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
    ],
    "model_params": {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "logistic_regression": {
            "C": 1.0
        },
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.1
        }
    }
}
