"""
Data Preprocessing Module for Movie Recommendation System

This module handles loading, cleaning, and preprocessing of the MovieLens dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os
from pathlib import Path

class DataPreprocessor:
    """
    Handles all data preprocessing tasks for the movie recommendation system.
    """
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_path: Path to the raw data directory
        """
        self.data_path = Path(data_path)
        self.ratings_df = None
        self.movies_df = None
        self.processed_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ratings and movies data from CSV files.
        
        Returns:
            Tuple of (ratings_df, movies_df)
        """
        try:
            # Load ratings data
            ratings_path = self.data_path / "ratings.csv"
            if not ratings_path.exists():
                raise FileNotFoundError(f"ratings.csv not found in {self.data_path}")
                
            self.ratings_df = pd.read_csv(ratings_path)
            print(f"Loaded {len(self.ratings_df):,} ratings")
            
            # Load movies data
            movies_path = self.data_path / "movies.csv"
            if not movies_path.exists():
                raise FileNotFoundError(f"movies.csv not found in {self.data_path}")
                
            self.movies_df = pd.read_csv(movies_path)
            print(f"Loaded {len(self.movies_df):,} movies")
            
            return self.ratings_df, self.movies_df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def validate_data(self) -> Dict[str, bool]:
        """
        Validate the loaded data for consistency and completeness.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {}
        
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Check required columns
        required_rating_cols = ['userId', 'movieId', 'rating']
        required_movie_cols = ['movieId', 'title']
        
        validation_results['ratings_columns'] = all(
            col in self.ratings_df.columns for col in required_rating_cols
        )
        validation_results['movies_columns'] = all(
            col in self.movies_df.columns for col in required_movie_cols
        )
        
        # Check for missing values
        validation_results['ratings_no_nulls'] = not self.ratings_df[required_rating_cols].isnull().any().any()
        validation_results['movies_no_nulls'] = not self.movies_df[required_movie_cols].isnull().any().any()
        
        # Check rating range
        validation_results['rating_range'] = (
            self.ratings_df['rating'].min() >= 0.5 and 
            self.ratings_df['rating'].max() <= 5.0
        )
        
        # Check movie ID consistency
        movie_ids_in_ratings = set(self.ratings_df['movieId'].unique())
        movie_ids_in_movies = set(self.movies_df['movieId'].unique())
        validation_results['movie_id_consistency'] = movie_ids_in_ratings.issubset(movie_ids_in_movies)
        
        return validation_results
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Returns:
            Cleaned and merged DataFrame
        """
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Merge ratings with movie information
        merged_df = self.ratings_df.merge(self.movies_df, on='movieId', how='left')
        
        # Remove any rows with missing movie information
        merged_df = merged_df.dropna(subset=['title'])
        
        # Sort by userId and timestamp if available
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.sort_values(['userId', 'timestamp'])
        else:
            merged_df = merged_df.sort_values(['userId', 'movieId'])
        
        # Reset index
        merged_df = merged_df.reset_index(drop=True)
        
        self.processed_df = merged_df
        return merged_df
    
    def get_data_statistics(self) -> Dict[str, any]:
        """
        Generate comprehensive statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        stats = {}
        
        # Basic statistics
        stats['total_ratings'] = len(self.processed_df)
        stats['unique_users'] = self.processed_df['userId'].nunique()
        stats['unique_movies'] = self.processed_df['movieId'].nunique()
        
        # Rating statistics
        stats['rating_mean'] = self.processed_df['rating'].mean()
        stats['rating_std'] = self.processed_df['rating'].std()
        stats['rating_distribution'] = self.processed_df['rating'].value_counts().sort_index().to_dict()
        
        # Sparsity
        total_possible_ratings = stats['unique_users'] * stats['unique_movies']
        stats['sparsity'] = 1 - (stats['total_ratings'] / total_possible_ratings)
        
        # User activity statistics
        user_activity = self.processed_df.groupby('userId').size()
        stats['avg_ratings_per_user'] = user_activity.mean()
        stats['median_ratings_per_user'] = user_activity.median()
        stats['min_ratings_per_user'] = user_activity.min()
        stats['max_ratings_per_user'] = user_activity.max()
        
        # Movie popularity statistics
        movie_popularity = self.processed_df.groupby('movieId').size()
        stats['avg_ratings_per_movie'] = movie_popularity.mean()
        stats['median_ratings_per_movie'] = movie_popularity.median()
        stats['min_ratings_per_movie'] = movie_popularity.min()
        stats['max_ratings_per_movie'] = movie_popularity.max()
        
        return stats
    
    def create_user_item_matrix(self, min_user_ratings: int = 20, min_movie_ratings: int = 20) -> pd.DataFrame:
        """
        Create a user-item matrix for collaborative filtering.
        
        Args:
            min_user_ratings: Minimum number of ratings a user must have
            min_movie_ratings: Minimum number of ratings a movie must have
            
        Returns:
            User-item matrix as DataFrame
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        # Filter users and movies with minimum number of ratings
        user_counts = self.processed_df['userId'].value_counts()
        movie_counts = self.processed_df['movieId'].value_counts()
        
        active_users = user_counts[user_counts >= min_user_ratings].index
        popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
        
        filtered_df = self.processed_df[
            (self.processed_df['userId'].isin(active_users)) &
            (self.processed_df['movieId'].isin(popular_movies))
        ]
        
        # Create user-item matrix
        user_item_matrix = filtered_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        print(f"User-item matrix shape: {user_item_matrix.shape}")
        print(f"Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
        
        return user_item_matrix
    
    def save_processed_data(self, output_path: str = "data/processed"):
        """
        Save processed data to files.
        
        Args:
            output_path: Path to save processed data
        """
        if self.processed_df is None:
            raise ValueError("Data not processed. Call clean_data() first.")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed DataFrame
        self.processed_df.to_csv(output_dir / "processed_ratings.csv", index=False)
        
        # Save statistics
        stats = self.get_data_statistics()
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(output_dir / "data_statistics.csv", index=False)
        
        print(f"Processed data saved to {output_dir}")

def main():
    """
    Example usage of the DataPreprocessor.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    try:
        # Load data
        ratings_df, movies_df = preprocessor.load_data()
        
        # Validate data
        validation_results = preprocessor.validate_data()
        print("Validation Results:", validation_results)
        
        # Clean data
        processed_df = preprocessor.clean_data()
        
        # Get statistics
        stats = preprocessor.get_data_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Create user-item matrix
        user_item_matrix = preprocessor.create_user_item_matrix()
        
        # Save processed data
        preprocessor.save_processed_data()
        
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please download the MovieLens 20M dataset and place the files in data/raw/")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
