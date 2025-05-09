# Smart Factory Energy Prediction Pipeline

## Overview
This pipeline predicts equipment energy consumption in a smart factory environment using machine learning. It leverages XGBoost regression and advanced feature engineering techniques to create accurate energy consumption forecasts based on environmental conditions, time patterns, and historical data.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Workflow](#pipeline-workflow)
- [Features](#features)
- [Model Performance](#model-performance)
- [Output Artifacts](#output-artifacts)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- shap
- matplotlib
- seaborn
- joblib

## Installation
Install required packages:

```bash
pip install pandas numpy scikit-learn xgboost optuna shap matplotlib seaborn joblib
```

## Usage
1. Prepare your dataset in CSV format with a timestamp column and equipment energy consumption data.
2. Update the file path in the script:
   ```python
   file_path = "path/to/your/data.csv"
   ```
3. Run the pipeline:
   ```python
   python energy_prediction_pipeline.py
   ```

## Pipeline Workflow
The pipeline consists of the following steps:

1. **Data Loading & Preprocessing**
   - Parses timestamps
   - Converts string columns to numeric
   - Handles missing values
   - Reports data quality issues

2. **Feature Engineering**
   - Creates time-based features (hour, day of week, seasonal indicators)
   - Generates zone temperature and humidity statistics
   - Calculates temperature differences between zones
   - Creates energy consumption lag features
   - Incorporates rolling window statistics
   - Adds feature interactions

3. **Target Transformation**
   - Log-transforms energy consumption for better model performance
   - Removes invalid or extreme energy values

4. **Feature Selection**
   - Removes low-variance features
   - Uses XGBoost importance to select most relevant features

5. **Dimensionality Reduction**
   - Applies PCA to zone-related features
   - Reduces multicollinearity

6. **Hyperparameter Optimization**
   - Uses Optuna for Bayesian optimization of XGBoost parameters
   - Implements time series cross-validation

7. **Model Training**
   - Trains XGBoost with optimal parameters
   - Includes fallback mechanisms for training errors

8. **Model Evaluation**
   - Calculates RMSE, MAE, R² in both original and log scales
   - Reports MAPE and median errors
   - Generates visualization of actual vs. predicted values

9. **Model Explainability**
   - Uses SHAP values to explain predictions
   - Creates feature importance visualizations

10. **Artifact Storage**
    - Saves trained model
    - Stores preprocessing components (scaler, feature selector)
    - Preserves visualization outputs

## Features
- **Robust Preprocessing**: Handles missing values and outliers
- **Comprehensive Feature Engineering**: Time-based, zone-based, and interaction features
- **Advanced Model Selection**: Hyperparameter optimization with time series validation
- **Explainable AI**: SHAP analysis for transparency in predictions
- **Error Handling**: Fallback mechanisms for training failures
- **Visual Analysis**: Multiple visualization outputs for model evaluation

## Model Performance
The pipeline outputs several performance metrics:

- **RMSE**: Root Mean Square Error (original and log scale)
- **MAE**: Mean Absolute Error (original and log scale)
- **R²**: Coefficient of determination (original and log scale)
- **MAPE**: Mean Absolute Percentage Error
- **Median Error**: Median of absolute errors

## Output Artifacts
The pipeline saves the following artifacts:

- `energy_prediction_model.joblib`: Trained XGBoost model
- `scaler.joblib`: Feature scaler
- `feature_selector.joblib`: Feature selection model
- `variance_selector.joblib`: Variance threshold selector
- `pca_model.joblib`: PCA model (if applicable)
- `feature_importance.png`: Feature importance visualization
- `shap_summary.png`: SHAP summary plot
- `actual_vs_predicted.png`: Visualization of predictions vs. actual values

## Customization
You can customize the pipeline by modifying these parameters:

- **Feature Engineering**: Add or remove engineered features in the `engineer_features` function
- **Feature Selection**: Adjust variance and importance thresholds in `select_features`
- **PCA Components**: Change `n_components` in `apply_pca_to_zones`
- **Hyperparameter Search**: Modify `n_trials` or parameter ranges in `optimize_hyperparameters`
- **Train/Test Split**: Change the test size ratio in the `main` function

## Troubleshooting

### Common Issues

1. **NaN or Infinite Values**
   - The pipeline has built-in checks for NaN/infinite values
   - Check your data preprocessing if errors persist

2. **Memory Issues**
   - For large datasets, reduce the number of engineered features
   - Consider sampling data for hyperparameter optimization

3. **Training Errors**
   - The pipeline includes fallback parameters if optimal ones fail
   - Check log for specific error messages

4. **Poor Performance**
   - Try increasing `n_trials` for hyperparameter optimization
   - Add more domain-specific features to the feature engineering step
   - Consider different time windows for lag features

5. **Execution Time**
   - Reduce `n_trials` for faster hyperparameter search
   - Use a smaller subset of data for development

### Debugging Tips
- Set `warnings.filterwarnings("default")` to see warnings
- Add print statements to track progress through large datasets
- Check feature distributions before and after transformations