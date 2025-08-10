"""
Feature Engineering for Machine Learning Models

This module creates intelligent features for predicting movie ratings
using machine learning algorithms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
from scipy import stats

class FeatureEngineer:
    """
    Creates features for machine learning models in movie recommendation system.
    """
    
    def __init__(self, ratings_df: pd.DataFrame):
        """
        Initialize the feature engineer.
        
        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating, ...]
        """
        self.ratings_df = ratings_df.copy()
        self.features_df = None
        
        # Calculate base statistics
        self._calculate_base_statistics()
    
    def _calculate_base_statistics(self):
        """Calculate basic statistics needed for feature engineering."""
        # Global statistics
        self.global_mean = self.ratings_df['rating'].mean()
        self.global_std = self.ratings_df['rating'].std()
        
        # User statistics
        self.user_stats = self.ratings_df.groupby('userId')['rating'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).add_prefix('user_')
        
        # Movie statistics
        self.movie_stats = self.ratings_df.groupby('movieId')['rating'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).add_prefix('movie_')
        
        print(f"Calculated statistics for {len(self.user_stats)} users and {len(self.movie_stats)} movies")
    
    def create_basic_features(self) -> pd.DataFrame:
        """
        Create basic features for each rating.
        
        Returns:
            DataFrame with basic features
        """
        print("Creating basic features...")
        
        # Start with original data
        features_df = self.ratings_df.copy()
        
        # Add user statistics
        features_df = features_df.merge(self.user_stats, left_on='userId', right_index=True, how='left')
        
        # Add movie statistics
        features_df = features_df.merge(self.movie_stats, left_on='movieId', right_index=True, how='left')
        
        # Global mean feature
        features_df['rating_gmean'] = self.global_mean
        
        # User bias (difference from global mean)
        features_df['user_bias'] = features_df['user_mean'] - self.global_mean
        
        # Movie bias (difference from global mean)
        features_df['movie_bias'] = features_df['movie_mean'] - self.global_mean
        
        # User rating standard deviation (rating diversity)
        features_df['user_std'] = features_df['user_std'].fillna(0)
        features_df['movie_std'] = features_df['movie_std'].fillna(0)
        
        print(f"Created basic features: {features_df.shape}")
        return features_df
    
    def create_advanced_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features based on user and movie interactions.
        
        Args:
            features_df: DataFrame with basic features
            
        Returns:
            DataFrame with advanced features added
        """
        print("Creating advanced features...")
        
        # User activity features
        features_df['user_activity_percentile'] = features_df['user_count'].rank(pct=True)
        features_df['is_heavy_user'] = (features_df['user_count'] > features_df['user_count'].quantile(0.8)).astype(int)
        features_df['is_light_user'] = (features_df['user_count'] < features_df['user_count'].quantile(0.2)).astype(int)
        
        # Movie popularity features
        features_df['movie_popularity_percentile'] = features_df['movie_count'].rank(pct=True)
        features_df['is_popular_movie'] = (features_df['movie_count'] > features_df['movie_count'].quantile(0.8)).astype(int)
        features_df['is_niche_movie'] = (features_df['movie_count'] < features_df['movie_count'].quantile(0.2)).astype(int)
        
        # Rating range features
        features_df['user_rating_range'] = features_df['user_max'] - features_df['user_min']
        features_df['movie_rating_range'] = features_df['movie_max'] - features_df['movie_min']
        
        # Z-score features (how unusual is this rating for user/movie)
        features_df['user_rating_zscore'] = (features_df['rating'] - features_df['user_mean']) / (features_df['user_std'] + 1e-8)
        features_df['movie_rating_zscore'] = (features_df['rating'] - features_df['movie_mean']) / (features_df['movie_std'] + 1e-8)
        
        # Interaction features
        features_df['user_movie_bias_interaction'] = features_df['user_bias'] * features_df['movie_bias']
        features_df['user_activity_movie_popularity'] = features_df['user_count'] * features_df['movie_count']
        
        # Rating deviation features
        features_df['rating_deviation_from_user_mean'] = features_df['rating'] - features_df['user_mean']
        features_df['rating_deviation_from_movie_mean'] = features_df['rating'] - features_df['movie_mean']
        features_df['rating_deviation_from_global_mean'] = features_df['rating'] - features_df['rating_gmean']
        
        print(f"Created advanced features: {features_df.shape}")
        return features_df
    
    def create_temporal_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features if timestamp information is available.
        
        Args:
            features_df: DataFrame with existing features
            
        Returns:
            DataFrame with temporal features added
        """
        if 'timestamp' not in features_df.columns:
            print("No timestamp column found, skipping temporal features")
            return features_df
        
        print("Creating temporal features...")
        
        # Convert timestamp to datetime
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'], unit='s')
        
        # Extract temporal components
        features_df['year'] = features_df['timestamp'].dt.year
        features_df['month'] = features_df['timestamp'].dt.month
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['hour'] = features_df['timestamp'].dt.hour
        
        # Rating recency features
        max_timestamp = features_df['timestamp'].max()
        features_df['days_since_rating'] = (max_timestamp - features_df['timestamp']).dt.days
        
        # User temporal patterns
        user_temporal = features_df.groupby('userId').agg({
            'timestamp': ['min', 'max', 'count'],
            'days_since_rating': 'mean'
        })
        user_temporal.columns = ['user_first_rating', 'user_last_rating', 'user_rating_count_temporal', 'user_avg_days_since']
        
        # User activity span
        user_temporal['user_activity_span_days'] = (user_temporal['user_last_rating'] - user_temporal['user_first_rating']).dt.days
        
        features_df = features_df.merge(user_temporal, left_on='userId', right_index=True, how='left')
        
        # Movie temporal patterns
        movie_temporal = features_df.groupby('movieId').agg({
            'timestamp': ['min', 'max', 'count'],
            'days_since_rating': 'mean'
        })
        movie_temporal.columns = ['movie_first_rating', 'movie_last_rating', 'movie_rating_count_temporal', 'movie_avg_days_since']
        
        features_df = features_df.merge(movie_temporal, left_on='movieId', right_index=True, how='left')
        
        print(f"Created temporal features: {features_df.shape}")
        return features_df
    
    def create_genre_features(self, features_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create genre-based features if movie genre information is available.
        
        Args:
            features_df: DataFrame with existing features
            movies_df: DataFrame with movie information including genres
            
        Returns:
            DataFrame with genre features added
        """
        if 'genres' not in movies_df.columns:
            print("No genres column found in movies data, skipping genre features")
            return features_df
        
        print("Creating genre features...")
        
        # Merge with movie information
        features_df = features_df.merge(movies_df[['movieId', 'genres']], on='movieId', how='left')
        
        # Parse genres (assuming pipe-separated format like "Action|Adventure|Sci-Fi")
        all_genres = set()
        for genres_str in movies_df['genres'].dropna():
            if genres_str != '(no genres listed)':
                all_genres.update(genres_str.split('|'))
        
        # Create binary genre features
        for genre in sorted(all_genres):
            features_df[f'genre_{genre.lower().replace("-", "_")}'] = features_df['genres'].str.contains(genre, na=False).astype(int)
        
        # Genre count feature
        features_df['genre_count'] = features_df['genres'].apply(
            lambda x: len(x.split('|')) if pd.notna(x) and x != '(no genres listed)' else 0
        )
        
        # User genre preferences
        genre_columns = [col for col in features_df.columns if col.startswith('genre_') and col != 'genre_count']
        
        for genre_col in genre_columns:
            user_genre_pref = features_df.groupby('userId')[genre_col].mean()
            features_df[f'user_pref_{genre_col}'] = features_df['userId'].map(user_genre_pref)
        
        print(f"Created genre features: {features_df.shape}")
        return features_df
    
    def create_collaborative_features(self, features_df: pd.DataFrame, k_similar: int = 50) -> pd.DataFrame:
        """
        Create features based on similar users and movies.
        
        Args:
            features_df: DataFrame with existing features
            k_similar: Number of similar users/movies to consider
            
        Returns:
            DataFrame with collaborative features added
        """
        print("Creating collaborative features...")
        
        # This is a simplified version - in practice, you'd use pre-computed similarity matrices
        
        # User similarity based features (simplified)
        user_rating_profiles = features_df.groupby('userId')['rating'].mean()
        
        # For each user, find average rating of similar users (simplified as users with similar average ratings)
        def get_similar_users_avg_rating(user_id, user_mean_rating):
            similar_users = user_rating_profiles[
                (user_rating_profiles >= user_mean_rating - 0.5) & 
                (user_rating_profiles <= user_mean_rating + 0.5) &
                (user_rating_profiles.index != user_id)
            ]
            return similar_users.mean() if len(similar_users) > 0 else user_mean_rating
        
        features_df['similar_users_avg_rating'] = features_df.apply(
            lambda row: get_similar_users_avg_rating(row['userId'], row['user_mean']), axis=1
        )
        
        # Movie similarity based features (simplified)
        movie_rating_profiles = features_df.groupby('movieId')['rating'].mean()
        
        def get_similar_movies_avg_rating(movie_id, movie_mean_rating):
            similar_movies = movie_rating_profiles[
                (movie_rating_profiles >= movie_mean_rating - 0.5) & 
                (movie_rating_profiles <= movie_mean_rating + 0.5) &
                (movie_rating_profiles.index != movie_id)
            ]
            return similar_movies.mean() if len(similar_movies) > 0 else movie_mean_rating
        
        features_df['similar_movies_avg_rating'] = features_df.apply(
            lambda row: get_similar_movies_avg_rating(row['movieId'], row['movie_mean']), axis=1
        )
        
        print(f"Created collaborative features: {features_df.shape}")
        return features_df
    
    def create_all_features(self, movies_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create all features in one go.
        
        Args:
            movies_df: Optional DataFrame with movie information
            
        Returns:
            Complete feature DataFrame
        """
        print("Creating all features...")
        
        # Create features step by step
        features_df = self.create_basic_features()
        features_df = self.create_advanced_features(features_df)
        features_df = self.create_temporal_features(features_df)
        
        if movies_df is not None:
            features_df = self.create_genre_features(features_df, movies_df)
        
        features_df = self.create_collaborative_features(features_df)
        
        # Store the final features
        self.features_df = features_df
        
        print(f"Final feature set: {features_df.shape}")
        print(f"Feature columns: {list(features_df.columns)}")
        
        return features_df
    
    def get_feature_importance_analysis(self, target_col: str = 'rating') -> Dict[str, float]:
        """
        Analyze feature importance using correlation and mutual information.
        
        Args:
            target_col: Target column name
            
        Returns:
            Dictionary with feature importance scores
        """
        if self.features_df is None:
            raise ValueError("Features not created. Call create_all_features() first.")
        
        print("Analyzing feature importance...")
        
        # Select only numeric features
        numeric_features = self.features_df.select_dtypes(include=[np.number])
        
        # Remove target and ID columns
        feature_cols = [col for col in numeric_features.columns 
                       if col not in [target_col, 'userId', 'movieId', 'timestamp']]
        
        if len(feature_cols) == 0:
            return {}
        
        X = numeric_features[feature_cols]
        y = numeric_features[target_col]
        
        # Calculate correlations
        correlations = {}
        for col in feature_cols:
            if X[col].std() > 0:  # Avoid constant features
                corr = X[col].corr(y)
                correlations[col] = abs(corr) if not np.isnan(corr) else 0
            else:
                correlations[col] = 0
        
        return correlations
    
    def save_features(self, output_path: str = "data/processed"):
        """
        Save the engineered features to disk.
        
        Args:
            output_path: Directory to save features
        """
        if self.features_df is None:
            raise ValueError("Features not created. Call create_all_features() first.")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        self.features_df.to_csv(output_dir / "engineered_features.csv", index=False)
        
        # Save feature importance
        feature_importance = self.get_feature_importance_analysis()
        importance_df = pd.DataFrame([feature_importance]).T
        importance_df.columns = ['importance_score']
        importance_df.to_csv(output_dir / "feature_importance.csv")
        
        print(f"Features saved to {output_dir}")

def main():
    """
    Example usage of the FeatureEngineer.
    """
    print("FeatureEngineer example usage")
    print("Note: This requires preprocessed ratings data")
    
    # Create dummy data for demonstration
    np.random.seed(42)
    n_users, n_movies, n_ratings = 1000, 500, 10000
    
    # Generate dummy ratings data
    user_ids = np.random.choice(range(1, n_users + 1), n_ratings)
    movie_ids = np.random.choice(range(1, n_movies + 1), n_ratings)
    ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    timestamps = np.random.randint(1000000000, 1600000000, n_ratings)  # Random timestamps
    
    ratings_df = pd.DataFrame({
        'userId': user_ids,
        'movieId': movie_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # Remove duplicates
    ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'])
    
    print(f"Created dummy ratings data: {ratings_df.shape}")
    
    # Create dummy movies data with genres
    genres_list = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    movies_data = []
    
    for movie_id in range(1, n_movies + 1):
        n_genres = np.random.randint(1, 4)  # 1-3 genres per movie
        movie_genres = np.random.choice(genres_list, n_genres, replace=False)
        movies_data.append({
            'movieId': movie_id,
            'title': f'Movie_{movie_id}',
            'genres': '|'.join(movie_genres)
        })
    
    movies_df = pd.DataFrame(movies_data)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(ratings_df)
    
    # Create all features
    features_df = feature_engineer.create_all_features(movies_df)
    
    # Analyze feature importance
    feature_importance = feature_engineer.get_feature_importance_analysis()
    
    print("\nTop 10 most important features:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"  {feature}: {importance:.4f}")

if __name__ == "__main__":
    main()
