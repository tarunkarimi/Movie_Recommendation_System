"""
Collaborative Filtering Implementation

This module implements user-based and item-based collaborative filtering
algorithms for movie recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle

class CollaborativeFiltering:
    """
    Implements collaborative filtering algorithms for movie recommendations.
    """
    
    def __init__(self, user_item_matrix: pd.DataFrame, 
                 movie_similarity_matrix: Optional[pd.DataFrame] = None,
                 user_similarity_matrix: Optional[pd.DataFrame] = None):
        """
        Initialize the collaborative filtering system.
        
        Args:
            user_item_matrix: User-item rating matrix
            movie_similarity_matrix: Pre-calculated movie similarity matrix
            user_similarity_matrix: Pre-calculated user similarity matrix
        """
        self.user_item_matrix = user_item_matrix
        self.movie_similarity_matrix = movie_similarity_matrix
        self.user_similarity_matrix = user_similarity_matrix
        
        # Calculate global and per-item/user means
        self.global_mean = self._calculate_global_mean()
        self.user_means = self._calculate_user_means()
        self.movie_means = self._calculate_movie_means()
    
    def _calculate_global_mean(self) -> float:
        """Calculate global mean rating across all users and movies."""
        non_zero_ratings = self.user_item_matrix.replace(0, np.nan)
        return non_zero_ratings.stack().mean()
    
    def _calculate_user_means(self) -> pd.Series:
        """Calculate mean rating for each user."""
        user_ratings = self.user_item_matrix.replace(0, np.nan)
        return user_ratings.mean(axis=1)
    
    def _calculate_movie_means(self) -> pd.Series:
        """Calculate mean rating for each movie."""
        movie_ratings = self.user_item_matrix.replace(0, np.nan)
        return movie_ratings.mean(axis=0)
    
    def predict_rating_item_based(self, user_id: int, movie_id: int, 
                                 k_neighbors: int = 50) -> float:
        """
        Predict rating using item-based collaborative filtering.
        
        Args:
            user_id: Target user ID
            movie_id: Target movie ID
            k_neighbors: Number of similar movies to consider
            
        Returns:
            Predicted rating
        """
        if self.movie_similarity_matrix is None:
            raise ValueError("Movie similarity matrix not provided")
        
        if user_id not in self.user_item_matrix.index:
            return self.movie_means.get(movie_id, self.global_mean)
        
        if movie_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.global_mean)
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0]
        
        if len(rated_movies) == 0:
            return self.movie_means.get(movie_id, self.global_mean)
        
        # Get similarities to target movie
        if movie_id not in self.movie_similarity_matrix.index:
            return self.movie_means.get(movie_id, self.global_mean)
        
        movie_similarities = self.movie_similarity_matrix.loc[movie_id]
        
        # Filter to movies the user has rated and are similar
        common_movies = set(rated_movies.index) & set(movie_similarities.index)
        if len(common_movies) == 0:
            return self.movie_means.get(movie_id, self.global_mean)
        
        # Get k most similar movies that user has rated
        relevant_similarities = movie_similarities[list(common_movies)]
        top_k_movies = relevant_similarities.nlargest(k_neighbors)
        
        if len(top_k_movies) == 0 or top_k_movies.sum() == 0:
            return self.movie_means.get(movie_id, self.global_mean)
        
        # Calculate weighted average
        numerator = sum(similarity * rated_movies[movie] 
                       for movie, similarity in top_k_movies.items())
        denominator = sum(abs(similarity) for similarity in top_k_movies.values())
        
        if denominator == 0:
            return self.movie_means.get(movie_id, self.global_mean)
        
        predicted_rating = numerator / denominator
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, predicted_rating))
    
    def predict_rating_user_based(self, user_id: int, movie_id: int, 
                                 k_neighbors: int = 50) -> float:
        """
        Predict rating using user-based collaborative filtering.
        
        Args:
            user_id: Target user ID
            movie_id: Target movie ID
            k_neighbors: Number of similar users to consider
            
        Returns:
            Predicted rating
        """
        if self.user_similarity_matrix is None:
            raise ValueError("User similarity matrix not provided")
        
        if user_id not in self.user_item_matrix.index:
            return self.movie_means.get(movie_id, self.global_mean)
        
        if movie_id not in self.user_item_matrix.columns:
            return self.user_means.get(user_id, self.global_mean)
        
        # Get movie's ratings from all users
        movie_ratings = self.user_item_matrix[movie_id]
        users_who_rated = movie_ratings[movie_ratings > 0]
        
        if len(users_who_rated) == 0:
            return self.user_means.get(user_id, self.global_mean)
        
        # Get similarities to target user
        if user_id not in self.user_similarity_matrix.index:
            return self.movie_means.get(movie_id, self.global_mean)
        
        user_similarities = self.user_similarity_matrix.loc[user_id]
        
        # Filter to users who rated the movie and are similar
        common_users = set(users_who_rated.index) & set(user_similarities.index)
        if len(common_users) == 0:
            return self.movie_means.get(movie_id, self.global_mean)
        
        # Get k most similar users who rated the movie
        relevant_similarities = user_similarities[list(common_users)]
        top_k_users = relevant_similarities.nlargest(k_neighbors)
        
        if len(top_k_users) == 0 or top_k_users.sum() == 0:
            return self.movie_means.get(movie_id, self.global_mean)
        
        # Calculate weighted average using mean-centered ratings
        target_user_mean = self.user_means.get(user_id, self.global_mean)
        
        numerator = 0
        denominator = 0
        
        for similar_user_id, similarity in top_k_users.items():
            similar_user_mean = self.user_means.get(similar_user_id, self.global_mean)
            user_rating = users_who_rated[similar_user_id]
            
            # Mean-centered rating
            centered_rating = user_rating - similar_user_mean
            
            numerator += similarity * centered_rating
            denominator += abs(similarity)
        
        if denominator == 0:
            return self.movie_means.get(movie_id, self.global_mean)
        
        predicted_rating = target_user_mean + (numerator / denominator)
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, predicted_rating))
    
    def recommend_movies_item_based(self, user_id: int, n_recommendations: int = 10,
                                   exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend movies using item-based collaborative filtering.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of movies to recommend
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_item_matrix.index:
            # Return most popular movies for new users
            movie_popularity = (self.user_item_matrix > 0).sum(axis=0)
            top_movies = movie_popularity.nlargest(n_recommendations)
            return [(movie_id, self.movie_means.get(movie_id, self.global_mean)) 
                   for movie_id in top_movies.index]
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        if exclude_rated:
            unrated_movies = user_ratings[user_ratings == 0].index
        else:
            unrated_movies = self.user_item_matrix.columns
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating_item_based(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def recommend_movies_user_based(self, user_id: int, n_recommendations: int = 10,
                                   exclude_rated: bool = True) -> List[Tuple[int, float]]:
        """
        Recommend movies using user-based collaborative filtering.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of movies to recommend
            exclude_rated: Whether to exclude already rated movies
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_item_matrix.index:
            # Return most popular movies for new users
            movie_popularity = (self.user_item_matrix > 0).sum(axis=0)
            top_movies = movie_popularity.nlargest(n_recommendations)
            return [(movie_id, self.movie_means.get(movie_id, self.global_mean)) 
                   for movie_id in top_movies.index]
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        if exclude_rated:
            unrated_movies = user_ratings[user_ratings == 0].index
        else:
            unrated_movies = self.user_item_matrix.columns
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating_user_based(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def evaluate_predictions(self, test_data: List[Tuple[int, int, float]], 
                           method: str = 'item_based') -> Dict[str, float]:
        """
        Evaluate prediction accuracy on test data.
        
        Args:
            test_data: List of (user_id, movie_id, actual_rating) tuples
            method: 'item_based' or 'user_based'
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        actuals = []
        
        for user_id, movie_id, actual_rating in test_data:
            if method == 'item_based':
                predicted_rating = self.predict_rating_item_based(user_id, movie_id)
            else:
                predicted_rating = self.predict_rating_user_based(user_id, movie_id)
            
            predictions.append(predicted_rating)
            actuals.append(actual_rating)
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'n_predictions': len(predictions)
        }

def main():
    """
    Example usage of the CollaborativeFiltering class.
    """
    print("CollaborativeFiltering example usage")
    print("Note: This requires preprocessed data and similarity matrices")
    
    # Example with dummy data (same as similarity_engine example)
    np.random.seed(42)
    n_users, n_movies = 100, 50
    
    # Create sparse user-item matrix
    user_ids = range(1, n_users + 1)
    movie_ids = range(1, n_movies + 1)
    
    user_item_matrix = pd.DataFrame(0, index=user_ids, columns=movie_ids)
    
    # Add some random ratings (sparse)
    for user in user_ids:
        n_ratings = np.random.randint(5, 20)
        movies_rated = np.random.choice(movie_ids, n_ratings, replace=False)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, 
                                 p=[0.1, 0.1, 0.2, 0.3, 0.3])
        
        for movie, rating in zip(movies_rated, ratings):
            user_item_matrix.loc[user, movie] = rating
    
    print(f"Created dummy user-item matrix: {user_item_matrix.shape}")
    
    # Create dummy similarity matrices (identity for simplicity)
    movie_similarity = pd.DataFrame(
        np.eye(n_movies) + np.random.normal(0, 0.1, (n_movies, n_movies)),
        index=movie_ids, columns=movie_ids
    )
    np.fill_diagonal(movie_similarity.values, 1.0)
    
    user_similarity = pd.DataFrame(
        np.eye(n_users) + np.random.normal(0, 0.1, (n_users, n_users)),
        index=user_ids, columns=user_ids
    )
    np.fill_diagonal(user_similarity.values, 1.0)
    
    # Initialize collaborative filtering
    cf = CollaborativeFiltering(
        user_item_matrix=user_item_matrix,
        movie_similarity_matrix=movie_similarity,
        user_similarity_matrix=user_similarity
    )
    
    # Test predictions
    test_user = 1
    test_movie = 5
    
    item_based_prediction = cf.predict_rating_item_based(test_user, test_movie)
    user_based_prediction = cf.predict_rating_user_based(test_user, test_movie)
    
    print(f"\nPredictions for User {test_user}, Movie {test_movie}:")
    print(f"Item-based: {item_based_prediction:.2f}")
    print(f"User-based: {user_based_prediction:.2f}")
    
    # Get recommendations
    item_recs = cf.recommend_movies_item_based(test_user, n_recommendations=5)
    user_recs = cf.recommend_movies_user_based(test_user, n_recommendations=5)
    
    print(f"\nItem-based recommendations for User {test_user}:")
    for movie_id, rating in item_recs:
        print(f"  Movie {movie_id}: {rating:.2f}")
    
    print(f"\nUser-based recommendations for User {test_user}:")
    for movie_id, rating in user_recs:
        print(f"  Movie {movie_id}: {rating:.2f}")

if __name__ == "__main__":
    main()
