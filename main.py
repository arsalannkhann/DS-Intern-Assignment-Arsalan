# Improved Energy Prediction Pipeline for Smart Factory

# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 2. Load Dataset
def load_and_preprocess_data(file_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Convert numeric-looking object columns to actual numbers
    numeric_cols = df.columns.drop('timestamp')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Print info about missing values
    missing_values = df.isnull().sum()
    print(f"Missing values before cleanup:\n{missing_values[missing_values > 0]}")
    
    # Handle missing values
    df.dropna(inplace=True)
    
    print(f"Data shape after preprocessing: {df.shape}")
    return df

# 3. Feature Engineering
def engineer_features(df):
    print("Engineering features...")
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_business_hours'] = df['hour'].between(8, 17).astype(int)
    df['is_morning'] = df['hour'].between(6, 9).astype(int)
    df['is_afternoon'] = df['hour'].between(12, 17).astype(int)
    df['is_evening'] = df['hour'].between(18, 22).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    
    # Sine and cosine transforms for cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # Zone-based features
    zone_temp_cols = [col for col in df.columns if 'zone' in col and 'temperature' in col]
    zone_hum_cols = [col for col in df.columns if 'zone' in col and 'humidity' in col]
    
    # Temperature statistics
    df['avg_zone_temp'] = df[zone_temp_cols].mean(axis=1)
    df['min_zone_temp'] = df[zone_temp_cols].min(axis=1)
    df['max_zone_temp'] = df[zone_temp_cols].max(axis=1)
    df['zone_temp_range'] = df['max_zone_temp'] - df['min_zone_temp']
    df['zone_temp_std'] = df[zone_temp_cols].std(axis=1)
    
    # Humidity statistics
    df['avg_zone_humidity'] = df[zone_hum_cols].mean(axis=1)
    df['min_zone_humidity'] = df[zone_hum_cols].min(axis=1)
    df['max_zone_humidity'] = df[zone_hum_cols].max(axis=1)
    df['zone_humidity_range'] = df['max_zone_humidity'] - df['min_zone_humidity']
    
    # Temperature differences between zones
    for i in range(1, len(zone_temp_cols)):
        for j in range(i+1, len(zone_temp_cols)+1):
            if f'zone{i}_temperature' in df.columns and f'zone{j}_temperature' in df.columns:
                df[f'temp_diff_zone{i}_zone{j}'] = df[f'zone{i}_temperature'] - df[f'zone{j}_temperature']
    
    # Energy-related lags and rolling statistics
    df['prev_energy'] = df['equipment_energy_consumption'].shift(1)
    df['energy_lag2'] = df['equipment_energy_consumption'].shift(2)
    df['energy_lag3'] = df['equipment_energy_consumption'].shift(3)
    df['energy_rolling_mean3'] = df['equipment_energy_consumption'].rolling(window=3).mean()
    df['energy_rolling_mean6'] = df['equipment_energy_consumption'].rolling(window=6).mean()
    df['energy_rolling_std3'] = df['equipment_energy_consumption'].rolling(window=3).std()
    
    # Pressure features
    df['atm_pressure_rolling_mean3'] = df['atmospheric_pressure'].rolling(window=3).mean()
    df['atm_pressure_rolling_std3'] = df['atmospheric_pressure'].rolling(window=3).std()
    
    # Ensure lighting_energy is treated as numeric
    if 'lighting_energy' in df.columns:
        df['lighting_energy'] = df['lighting_energy'].astype(int)
    
    # Interactions between features
    df['temp_pressure_interaction'] = df['avg_zone_temp'] * df['atmospheric_pressure']
    df['humidity_pressure_interaction'] = df['avg_zone_humidity'] * df['atmospheric_pressure']
    
    # Drop rows with NaN values created by lags and rolling calculations
    missing_count = df.isnull().sum().sum()
    print(f"Missing values after feature engineering: {missing_count}")
    df.dropna(inplace=True)
    print(f"Data shape after feature engineering: {df.shape}")
    
    return df

# 4. Target Transformation and Feature Preparation
def prepare_features_target(df):
    print("Preparing features and target...")
    
    # Check and handle extreme or invalid values in energy consumption
    # Filter out zero, negative, or extremely large values
    valid_energy = (df['equipment_energy_consumption'] > 0) & (df['equipment_energy_consumption'] < 1e10)
    
    if (~valid_energy).any():
        print(f"Removed {(~valid_energy).sum()} rows with invalid energy consumption values")
        df = df[valid_energy]
    
    # Log transform the target for better model performance
    df['equipment_energy_consumption_log'] = np.log1p(df['equipment_energy_consumption'])
    
    # Make sure there are no infinite or NaN values in the target
    if np.isinf(df['equipment_energy_consumption_log']).any() or np.isnan(df['equipment_energy_consumption_log']).any():
        print("Warning: Found infinite or NaN values in log-transformed target. Removing these rows.")
        df = df[~np.isinf(df['equipment_energy_consumption_log']) & ~np.isnan(df['equipment_energy_consumption_log'])]
    
    # Prepare features and target
    X = df.drop(['equipment_energy_consumption', 'equipment_energy_consumption_log', 'timestamp'], axis=1)
    y = df['equipment_energy_consumption_log']
    
    # Final check for any NaN or infinite values in X or y
    X_has_invalid = X.isna().any().any() or np.isinf(X.values).any()
    y_has_invalid = y.isna().any() or np.isinf(y.values).any()
    
    if X_has_invalid:
        print("Warning: Features still contain NaN or infinite values. Removing these rows.")
        valid_X_mask = ~X.isna().any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_X_mask]
        y = y[valid_X_mask]
    
    if y_has_invalid:
        print("Warning: Target still contains NaN or infinite values. Removing these rows.")
        valid_y_mask = ~y.isna() & ~np.isinf(y)
        X = X[valid_y_mask]
        y = y[valid_y_mask]
    
    print(f"Feature set shape after cleaning: {X.shape}")
    return X, y

# 5. Feature Selection
def select_features(X_train, y_train, threshold=0.01, importance_threshold=0.01):
    print("Performing feature selection...")
    
    # Remove low variance features
    selector = VarianceThreshold(threshold=threshold)
    X_train_var = selector.fit_transform(X_train)
    selected_features = X_train.columns[selector.get_support()]
    
    # Initial XGBoost for feature importance
    init_model = XGBRegressor(n_estimators=100, random_state=RANDOM_SEED)
    init_model.fit(X_train, y_train)
    
    # Use model-based feature selection
    feature_selector = SelectFromModel(init_model, threshold=importance_threshold, prefit=True)
    X_train_selected = feature_selector.transform(X_train[selected_features])
    final_features = selected_features[feature_selector.get_support()]
    
    print(f"Features reduced from {X_train.shape[1]} to {len(final_features)}")
    print(f"Selected features: {', '.join(final_features)}")
    
    return final_features, selector, feature_selector

# 6. Dimensionality Reduction for Zone Features
def apply_pca_to_zones(X_train, X_test, zone_features, n_components=5):
    print("Applying PCA to zone features...")
    
    if len(zone_features) > n_components:
        pca = PCA(n_components=n_components)
        zone_pca_train = pca.fit_transform(X_train[zone_features])
        zone_pca_test = pca.transform(X_test[zone_features])
        
        # Create DataFrames with PCA results
        zone_pca_train_df = pd.DataFrame(
            zone_pca_train, 
            columns=[f'pca_zone_{i}' for i in range(1, n_components+1)],
            index=X_train.index
        )
        zone_pca_test_df = pd.DataFrame(
            zone_pca_test, 
            columns=[f'pca_zone_{i}' for i in range(1, n_components+1)],
            index=X_test.index
        )
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        print(f"PCA explained variance: {explained_var}")
        print(f"Total explained variance: {np.sum(explained_var):.2f}")
        
        # Replace original zone features with PCA components
        X_train_pca = pd.concat([X_train.drop(columns=zone_features), zone_pca_train_df], axis=1)
        X_test_pca = pd.concat([X_test.drop(columns=zone_features), zone_pca_test_df], axis=1)
        
        return X_train_pca, X_test_pca, pca
    else:
        print("Not enough zone features for PCA, skipping...")
        return X_train, X_test, None

# 7. Hyperparameter Tuning with Optuna
def optimize_hyperparameters(X_train, y_train, n_trials=50):
    print(f"Optimizing hyperparameters with Optuna ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "alpha": trial.suggest_float("alpha", 0, 10),
            "lambda": trial.suggest_float("lambda", 0, 10),
            "random_state": RANDOM_SEED,
        }

        model = XGBRegressor(**params)
        
        n_splits = 5
        n_samples = len(X_train)

        # Calculate the maximum allowed test_size to avoid split errors
        max_test_size = n_samples // (n_splits + 1)

        # Sample test_size as a fraction and convert to integer count
        test_size_fraction = trial.suggest_float("test_size", 0.1, 0.4)
        test_size = int(test_size_fraction * n_samples)
        test_size = min(test_size, max_test_size)
        test_size = max(test_size, 1)  # ensure at least 1

        # Now use it in TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
    

        scores = []

        for train_idx, valid_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            
            # Add error handling for NaN/infinite values
            try:
                # Simple fit without eval_set
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                scores.append(rmse)
            except Exception as e:
                print(f"Error during training: {e}")
                # Return a poor score if training fails
                return float('inf')

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"Best RMSE: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    return best_params

# 8. Train Final Model
def train_final_model(X_train, X_test, y_train, y_test, best_params):
    print("Training final model with best parameters...")
    
    # Final check for NaN or infinite values
    X_train_clean = X_train.copy()
    y_train_clean = y_train.copy()
    
    # Check and remove any rows with NaN or inf values
    valid_rows = ~np.isnan(y_train_clean) & ~np.isinf(y_train_clean) & ~X_train_clean.isna().any(axis=1) & ~np.isinf(X_train_clean).any(axis=1)
    if (~valid_rows).any():
        print(f"Removed {(~valid_rows).sum()} invalid rows before final training")
        X_train_clean = X_train_clean[valid_rows]
        y_train_clean = y_train_clean[valid_rows]
    
    # Create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_clean, y_train_clean, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Add early_stopping_rounds parameter
    model = xgb.XGBRegressor(**best_params, random_state=RANDOM_SEED)
    
    # Train with validation set for early stopping
    try:
        # Simple train without eval_set or early stopping
        model.fit(X_train_final, y_train_final)
        
        # Save feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Important Features:")
        print(feature_importance.head(15))
        
        return model, feature_importance
    
    except Exception as e:
        print(f"Error during final model training: {e}")
        print("Attempting to train with more robust parameters...")
        
        # Fall back to more conservative parameters if best params fail
        fallback_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "alpha": 1.0,
            "lambda": 1.0,
            "random_state": RANDOM_SEED
        }
        
        model = xgb.XGBRegressor(**fallback_params)
        model.fit(X_train_final, y_train_final)
        
        # Save feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Important Features:")
        print(feature_importance.head(15))
        
        return model, feature_importance

# 9. Evaluate Model
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    
    # Make predictions
    y_pred_log = model.predict(X_test)
    
    # Transform back to original scale
    y_true = np.expm1(y_test)
    y_pred = np.expm1(y_pred_log)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Log scale metrics
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)
    r2_log = r2_score(y_test, y_pred_log)
    
    print("\nModel Evaluation (Original Scale):")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    print("\nModel Evaluation (Log Scale):")
    print(f"RMSE (log): {rmse_log:.4f}")
    print(f"MAE (log): {mae_log:.4f}")
    print(f"R² Score (log): {r2_log:.4f}")
    
    # Calculate percentage errors (with protection against divide by zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Calculate median errors
    median_error = np.median(np.abs(y_true - y_pred))
    print(f"Median Absolute Error: {median_error:.2f}")
    
    # Create dataframe for visualization
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': y_true - y_pred,
        'Percentage_Error': ((y_true - y_pred) / np.maximum(y_true, 1e-10)) * 100
    })
    
    return results_df, {
        'rmse': rmse, 'mae': mae, 'r2': r2, 
        'rmse_log': rmse_log, 'mae_log': mae_log, 'r2_log': r2_log,
        'mape': mape, 'median_error': median_error
    }

# 10. SHAP Analysis
def explain_model(model, X_train, X_test):
    print("Generating SHAP explanations...")
    
    try:
        # Create explainer
        explainer = shap.Explainer(model)
        
        # Calculate SHAP values for test set
        shap_values = explainer(X_test)
        
        print("SHAP Analysis Complete.")
        return explainer, shap_values
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        print("Skipping SHAP analysis...")
        return None, None

# 11. Save Model and Artifacts
def save_model_artifacts(model, scaler, feature_selector, var_selector, pca=None):
    print("Saving model and artifacts...")
    
    try:
        dump(model, 'energy_prediction_model.joblib')
        dump(scaler, 'scaler.joblib')
        dump(feature_selector, 'feature_selector.joblib')
        dump(var_selector, 'variance_selector.joblib')
        
        if pca is not None:
            dump(pca, 'pca_model.joblib')
        
        print("Model and artifacts saved successfully.")
    except Exception as e:
        print(f"Error saving artifacts: {e}")

# 12. Main Function
def main(file_path):
    try:
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        
        # Engineer features
        df = engineer_features(df)
        
        # Prepare features and target
        X, y = prepare_features_target(df)
        
        # Train-Test Split - Use TimeSeriesSplit for time series data
        # Keep temporal ordering by not shuffling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Feature Selection
        selected_features, var_selector, feature_selector = select_features(X_train, y_train)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        # PCA for zone features
        zone_features = [col for col in selected_features if 'zone' in col]
        if len(zone_features) > 5:
            X_train, X_test, pca_model = apply_pca_to_zones(X_train, X_test, zone_features)
        else:
            pca_model = None
        
        # Feature Scaling - Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Final check for any NaN or inf values before training
        X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], np.nan).dropna()
        y_train_clean = y_train.loc[X_train_scaled.index] 
        X_test_scaled = X_test_scaled.replace([np.inf, -np.inf], np.nan).dropna()
        y_test_clean = y_test.loc[X_test_scaled.index]
        
        # Hyperparameter Tuning
        best_params = optimize_hyperparameters(X_train_scaled, y_train_clean, n_trials=30)
        
        # Train Final Model
        model, feature_importance = train_final_model(
            X_train_scaled, X_test_scaled, y_train_clean, y_test_clean, best_params
        )
        
        # Evaluate Model
        results_df, metrics = evaluate_model(model, X_test_scaled, y_test_clean)
        
        # Generate SHAP Explanations
        explainer, shap_values = explain_model(model, X_train_scaled, X_test_scaled)
        
        # Save Model and Artifacts
        save_model_artifacts(model, scaler, feature_selector, var_selector, pca_model)
        
        # Feature importance visualization
        plt.figure(figsize=(12, 8))
        feature_importance.head(15).plot(kind='barh', x='Feature', y='Importance', legend=False)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # SHAP summary plot
        if explainer is not None and shap_values is not None:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_scaled, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png')
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(np.expm1(y_test_clean), np.expm1(model.predict(X_test_scaled)), alpha=0.5)
        plt.xlabel('Actual Energy Consumption')
        plt.ylabel('Predicted Energy Consumption')
        plt.title('Actual vs Predicted Energy Consumption')
        ideal_line = np.linspace(
            min(np.expm1(y_test_clean)), 
            max(np.expm1(y_test_clean)), 
            100
        )
        plt.plot(ideal_line, ideal_line, 'r--')
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        
        print("\nEnergy Prediction Pipeline completed successfully!")
        return model, metrics, feature_importance, results_df
        
    except Exception as e:
        print(f"Error in main pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    file_path = "/Users/arsalankhan/Desktop/DS-Intern-Assignment-Arsalan/data/data.csv"  # Update with your file path
    model, metrics, feature_importance, results = main(file_path)