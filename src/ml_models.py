"""
Machine Learning Models for Movie Rating Prediction

This module implements various ML models for predicting movie ratings,
including XGBoost, Random Forest, and others.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import json

# ML Libraries (will be imported when available)
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

class MLModelTrainer:
    """
    Trains and evaluates machine learning models for rating prediction.
    """
    
    def __init__(self, features_df: pd.DataFrame, target_col: str = 'rating'):
        """
        Initialize the ML model trainer.
        
        Args:
            features_df: DataFrame with engineered features
            target_col: Name of the target column
        """
        if not ML_AVAILABLE:
            raise ImportError("Required ML libraries not available. Install requirements.txt")
        
        self.features_df = features_df.copy()
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for training."""
        print("Preparing data for ML training...")
        
        # Select feature columns (exclude IDs and target)
        exclude_cols = [self.target_col, 'userId', 'movieId']
        if 'timestamp' in self.features_df.columns:
            exclude_cols.append('timestamp')
        
        # Select numeric features only
        numeric_df = self.features_df.select_dtypes(include=[np.number])
        self.feature_cols = [col for col in numeric_df.columns if col not in exclude_cols]
        
        # Handle missing values
        self.X = numeric_df[self.feature_cols].fillna(0)
        self.y = numeric_df[self.target_col]
        
        # Remove rows with missing target values
        valid_mask = ~self.y.isna()
        self.X = self.X[valid_mask]
        self.y = self.y[valid_mask]
        
        print(f"Prepared data: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"Feature columns: {self.feature_cols[:10]}...")  # Show first 10 features
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Split data - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
    
    def train_linear_models(self):
        """Train linear regression models."""
        print("Training linear models...")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1)
        }
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Use scaled features for linear models
            model.fit(self.X_train_scaled, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            
            # Store model and results
            self.models[name] = model
            self.results[name] = self._calculate_metrics(
                self.y_train, y_pred_train, self.y_test, y_pred_test
            )
    
    def train_tree_models(self):
        """Train tree-based models."""
        print("Training tree-based models...")
        
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Use original features for tree models
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            # Store model and results
            self.models[name] = model
            self.results[name] = self._calculate_metrics(
                self.y_train, y_pred_train, self.y_test, y_pred_test
            )
    
    def train_xgboost(self, hyperparameter_tuning: bool = False):
        """Train XGBoost model."""
        print("Training XGBoost model...")
        
        if hyperparameter_tuning:
            print("  Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            print(f"  Best parameters: {grid_search.best_params_}")
        else:
            best_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.9,
                random_state=42
            )
            best_model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = best_model.predict(self.X_train)
        y_pred_test = best_model.predict(self.X_test)
        
        # Store model and results
        self.models['XGBoost'] = best_model
        self.results['XGBoost'] = self._calculate_metrics(
            self.y_train, y_pred_train, self.y_test, y_pred_test
        )
    
    def train_lightgbm(self):
        """Train LightGBM model."""
        print("Training LightGBM model...")
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            random_state=42
            
        )
        
        model.fit(self.X_train, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        
        # Store model and results
        self.models['LightGBM'] = model
        self.results['LightGBM'] = self._calculate_metrics(
            self.y_train, y_pred_train, self.y_test, y_pred_test
        )
    
    def _calculate_metrics(self, y_train_true, y_train_pred, y_test_true, y_test_pred) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))
        
        def mape(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'Train_RMSE': rmse(y_train_true, y_train_pred),
            'Test_RMSE': rmse(y_test_true, y_test_pred),
            'Train_MAE': mean_absolute_error(y_train_true, y_train_pred),
            'Test_MAE': mean_absolute_error(y_test_true, y_test_pred),
            'Train_MAPE': mape(y_train_true, y_train_pred),
            'Test_MAPE': mape(y_test_true, y_test_pred)
        }
        
        return metrics
    
    def train_all_models(self, include_tuning: bool = False):
        """Train all available models."""
        print("Training all models...")
        
        # Split data first
        self.split_data()
        
        # Train models
        self.train_linear_models()
        self.train_tree_models()
        self.train_xgboost(hyperparameter_tuning=include_tuning)
        self.train_lightgbm()
        
        print("All models trained successfully!")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of all models."""
        if not self.results:
            raise ValueError("No models trained. Call train_all_models() first.")
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)
        
        # Sort by test RMSE (lower is better)
        comparison_df = comparison_df.sort_values('Test_RMSE')
        
        return comparison_df
    
    def get_feature_importance(self, model_name: str = 'XGBoost', top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from a tree-based model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            raise ValueError(f"Model {model_name} does not have feature importance")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def predict_rating(self, user_features: Dict[str, float], model_name: str = 'XGBoost') -> float:
        """Predict rating for new user-movie combination."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Create feature vector
        feature_vector = np.zeros(len(self.feature_cols))
        for i, feature in enumerate(self.feature_cols):
            if feature in user_features:
                feature_vector[i] = user_features[feature]
        
        # Scale if needed (for linear models)
        if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))
            prediction = model.predict(feature_vector)[0]
        else:
            prediction = model.predict(feature_vector.reshape(1, -1))[0]
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, prediction))
    
    def save_models(self, output_path: str = "models"):
        """Save trained models and results."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = output_dir / f"{name.lower().replace(' ', '_')}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        scaler_path = output_dir / "feature_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save results and metadata
        results_path = output_dir / "model_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        metadata = {
            'feature_columns': self.feature_cols,
            'target_column': self.target_col,
            'data_shape': list(self.X.shape)
        }
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models and results saved to {output_dir}")
    
    def cross_validate_model(self, model_name: str = 'XGBoost', cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, self.X, self.y, cv=cv_folds, 
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        
        cv_results = {
            'CV_RMSE_mean': -cv_scores.mean(),
            'CV_RMSE_std': cv_scores.std(),
            'CV_scores': -cv_scores
        }
        
        return cv_results

def main():
    """
    Example usage of the MLModelTrainer.
    """
    if not ML_AVAILABLE:
        print("ML libraries not available. Please install requirements.txt")
        return
    
    print("MLModelTrainer example usage")
    print("Note: This requires engineered features data")
    
    # Create dummy feature data
    np.random.seed(42)
    n_samples = 5000
    
    # Create dummy features similar to what FeatureEngineer would produce
    feature_data = {
        'userId': np.random.randint(1, 1000, n_samples),
        'movieId': np.random.randint(1, 500, n_samples),
        'user_mean': np.random.normal(3.5, 0.5, n_samples),
        'movie_mean': np.random.normal(3.5, 0.4, n_samples),
        'rating_gmean': np.full(n_samples, 3.5),
        'user_bias': np.random.normal(0, 0.3, n_samples),
        'movie_bias': np.random.normal(0, 0.25, n_samples),
        'user_count': np.random.randint(5, 200, n_samples),
        'movie_count': np.random.randint(10, 1000, n_samples),
        'user_std': np.random.uniform(0.5, 1.5, n_samples),
        'movie_std': np.random.uniform(0.3, 1.2, n_samples),
    }
    
    # Create target variable (rating) based on features with some noise
    ratings = (
        feature_data['user_mean'] * 0.4 +
        feature_data['movie_mean'] * 0.4 +
        feature_data['user_bias'] * 0.1 +
        feature_data['movie_bias'] * 0.1 +
        np.random.normal(0, 0.2, n_samples)
    )
    
    # Clamp to valid range
    ratings = np.clip(ratings, 0.5, 5.0)
    feature_data['rating'] = ratings
    
    features_df = pd.DataFrame(feature_data)
    
    print(f"Created dummy feature data: {features_df.shape}")
    
    # Initialize trainer
    trainer = MLModelTrainer(features_df)
    
    # Train all models
    trainer.train_all_models()
    
    # Get model comparison
    comparison = trainer.get_model_comparison()
    print("\nModel Comparison:")
    print(comparison)
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance('XGBoost', top_n=10)
    print("\nTop 10 Feature Importance (XGBoost):")
    print(feature_importance)
    
    # Example prediction
    sample_features = {
        'user_mean': 4.2,
        'movie_mean': 3.8,
        'rating_gmean': 3.5,
        'user_bias': 0.7,
        'movie_bias': 0.3,
        'user_count': 50,
        'movie_count': 200,
        'user_std': 1.0,
        'movie_std': 0.8
    }
    
    predicted_rating = trainer.predict_rating(sample_features, 'XGBoost')
    print(f"\nExample prediction: {predicted_rating:.2f}")

if __name__ == "__main__":
    main()
