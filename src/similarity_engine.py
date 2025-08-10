"""
Similarity Engine for Movie Recommendation System

This module implements movie-movie and user-user similarity calculations
using various similarity metrics.
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

class SimilarityEngine:
    """
    Calculates and manages similarity matrices for collaborative filtering.
    """
    
    def __init__(self, user_item_matrix: pd.DataFrame):
        """
        Initialize the similarity engine.
        
        Args:
            user_item_matrix: User-item matrix (users as rows, movies as columns)
        """
        self.user_item_matrix = user_item_matrix
        self.movie_similarity_matrix = None
        self.user_similarity_matrix = None
        
    def calculate_movie_similarity(self, method: str = 'cosine', min_common_users: int = 5) -> pd.DataFrame:
        """
        Calculate movie-to-movie similarity matrix.
        
        Args:
            method: Similarity method ('cosine', 'pearson', 'adjusted_cosine')
            min_common_users: Minimum number of common users for similarity calculation
            
        Returns:
            Movie similarity matrix
        """
        print(f"Calculating movie similarity using {method} method...")
        
        # Transpose to get movies as rows
        movie_matrix = self.user_item_matrix.T
        
        if method == 'cosine':
            # Replace 0s with NaN for proper cosine similarity
            movie_matrix_filled = movie_matrix.replace(0, np.nan)
            similarity_matrix = self._cosine_similarity_with_nan(movie_matrix_filled)
            
        elif method == 'pearson':
            similarity_matrix = self._pearson_similarity(movie_matrix, min_common_users)
            
        elif method == 'adjusted_cosine':
            # Mean-center by user ratings
            user_means = self.user_item_matrix.replace(0, np.nan).mean(axis=1)
            adjusted_matrix = self.user_item_matrix.sub(user_means, axis=0).fillna(0)
            similarity_matrix = self._cosine_similarity_with_nan(adjusted_matrix.T)
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=movie_matrix.index,
            columns=movie_matrix.index
        )
        
        self.movie_similarity_matrix = similarity_df
        print(f"Movie similarity matrix shape: {similarity_df.shape}")
        return similarity_df
    
    def calculate_user_similarity(self, method: str = 'cosine', min_common_movies: int = 5) -> pd.DataFrame:
        """
        Calculate user-to-user similarity matrix.
        
        Args:
            method: Similarity method ('cosine', 'pearson')
            min_common_movies: Minimum number of common movies for similarity calculation
            
        Returns:
            User similarity matrix
        """
        print(f"Calculating user similarity using {method} method...")
        
        if method == 'cosine':
            # Replace 0s with NaN for proper cosine similarity
            user_matrix_filled = self.user_item_matrix.replace(0, np.nan)
            similarity_matrix = self._cosine_similarity_with_nan(user_matrix_filled)
            
        elif method == 'pearson':
            similarity_matrix = self._pearson_similarity(self.user_item_matrix, min_common_movies)
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Convert to DataFrame
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        self.user_similarity_matrix = similarity_df
        print(f"User similarity matrix shape: {similarity_df.shape}")
        return similarity_df
    
    def _cosine_similarity_with_nan(self, matrix: pd.DataFrame) -> np.ndarray:
        """
        Calculate cosine similarity handling NaN values.
        
        Args:
            matrix: Input matrix with potential NaN values
            
        Returns:
            Similarity matrix as numpy array
        """
        # Fill NaN with 0 for cosine similarity calculation
        matrix_filled = matrix.fillna(0).values
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(matrix_filled)
        
        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _pearson_similarity(self, matrix: pd.DataFrame, min_common: int) -> np.ndarray:
        """
        Calculate Pearson correlation similarity.
        
        Args:
            matrix: Input matrix
            min_common: Minimum number of common rated items
            
        Returns:
            Similarity matrix as numpy array
        """
        n_items = len(matrix)
        similarity_matrix = np.eye(n_items)  # Initialize with identity matrix
        
        for i in range(n_items):
            for j in range(i + 1, n_items):
                # Get ratings for both items
                ratings_i = matrix.iloc[i].replace(0, np.nan)
                ratings_j = matrix.iloc[j].replace(0, np.nan)
                
                # Find common ratings
                common_mask = ~(ratings_i.isna() | ratings_j.isna())
                common_ratings_i = ratings_i[common_mask]
                common_ratings_j = ratings_j[common_mask]
                
                # Calculate correlation if enough common ratings
                if len(common_ratings_i) >= min_common:
                    try:
                        correlation, _ = pearsonr(common_ratings_i, common_ratings_j)
                        if not np.isnan(correlation):
                            similarity_matrix[i, j] = correlation
                            similarity_matrix[j, i] = correlation
                    except:
                        pass  # Keep default 0 similarity
        
        return similarity_matrix
    
    def get_similar_movies(self, movie_id: int, n_similar: int = 10, 
                          similarity_threshold: float = 0.1) -> List[Tuple[int, float, str]]:
        """
        Get most similar movies to a given movie.
        
        Args:
            movie_id: Target movie ID
            n_similar: Number of similar movies to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (movie_id, similarity_score, movie_title) tuples
        """
        if self.movie_similarity_matrix is None:
            raise ValueError("Movie similarity matrix not calculated. Call calculate_movie_similarity() first.")
        
        if movie_id not in self.movie_similarity_matrix.index:
            raise ValueError(f"Movie ID {movie_id} not found in similarity matrix.")
        
        # Get similarities for the target movie
        similarities = self.movie_similarity_matrix.loc[movie_id]
        
        # Filter by threshold and exclude self
        filtered_similarities = similarities[
            (similarities >= similarity_threshold) & 
            (similarities.index != movie_id)
        ]
        
        # Sort and get top N
        top_similar = filtered_similarities.nlargest(n_similar)
        
        # Get movie titles (this would need to be connected to movie metadata)
        similar_movies = []
        for movie_id_sim, similarity in top_similar.items():
            similar_movies.append((movie_id_sim, similarity, f"Movie_{movie_id_sim}"))
        
        return similar_movies
    
    def get_similar_users(self, user_id: int, n_similar: int = 10, 
                         similarity_threshold: float = 0.1) -> List[Tuple[int, float]]:
        """
        Get most similar users to a given user.
        
        Args:
            user_id: Target user ID
            n_similar: Number of similar users to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if self.user_similarity_matrix is None:
            raise ValueError("User similarity matrix not calculated. Call calculate_user_similarity() first.")
        
        if user_id not in self.user_similarity_matrix.index:
            raise ValueError(f"User ID {user_id} not found in similarity matrix.")
        
        # Get similarities for the target user
        similarities = self.user_similarity_matrix.loc[user_id]
        
        # Filter by threshold and exclude self
        filtered_similarities = similarities[
            (similarities >= similarity_threshold) & 
            (similarities.index != user_id)
        ]
        
        # Sort and get top N
        top_similar = filtered_similarities.nlargest(n_similar)
        
        return [(user_id, similarity) for user_id, similarity in top_similar.items()]
    
    def save_similarity_matrices(self, output_path: str = "models"):
        """
        Save calculated similarity matrices to disk.
        
        Args:
            output_path: Directory to save matrices
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.movie_similarity_matrix is not None:
            with open(output_dir / "movie_similarity_matrix.pkl", "wb") as f:
                pickle.dump(self.movie_similarity_matrix, f)
            print(f"Movie similarity matrix saved to {output_dir / 'movie_similarity_matrix.pkl'}")
        
        if self.user_similarity_matrix is not None:
            with open(output_dir / "user_similarity_matrix.pkl", "wb") as f:
                pickle.dump(self.user_similarity_matrix, f)
            print(f"User similarity matrix saved to {output_dir / 'user_similarity_matrix.pkl'}")
    
    def load_similarity_matrices(self, input_path: str = "models"):
        """
        Load similarity matrices from disk.
        
        Args:
            input_path: Directory containing saved matrices
        """
        input_dir = Path(input_path)
        
        movie_sim_path = input_dir / "movie_similarity_matrix.pkl"
        if movie_sim_path.exists():
            with open(movie_sim_path, "rb") as f:
                self.movie_similarity_matrix = pickle.load(f)
            print(f"Movie similarity matrix loaded from {movie_sim_path}")
        
        user_sim_path = input_dir / "user_similarity_matrix.pkl"
        if user_sim_path.exists():
            with open(user_sim_path, "rb") as f:
                self.user_similarity_matrix = pickle.load(f)
            print(f"User similarity matrix loaded from {user_sim_path}")

def main():
    """
    Example usage of the SimilarityEngine.
    """
    # This would typically load from preprocessed data
    print("SimilarityEngine example usage")
    print("Note: This requires preprocessed user-item matrix data")
    
    # Example with dummy data
    np.random.seed(42)
    n_users, n_movies = 100, 50
    
    # Create sparse user-item matrix
    user_ids = range(1, n_users + 1)
    movie_ids = range(1, n_movies + 1)
    
    user_item_matrix = pd.DataFrame(0, index=user_ids, columns=movie_ids)
    
    # Add some random ratings (sparse)
    for user in user_ids:
        n_ratings = np.random.randint(5, 20)  # Each user rates 5-20 movies
        movies_rated = np.random.choice(movie_ids, n_ratings, replace=False)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, 
                                 p=[0.1, 0.1, 0.2, 0.3, 0.3])  # Bias toward higher ratings
        
        for movie, rating in zip(movies_rated, ratings):
            user_item_matrix.loc[user, movie] = rating
    
    print(f"Created dummy user-item matrix: {user_item_matrix.shape}")
    print(f"Sparsity: {(user_item_matrix == 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1]):.4f}")
    
    # Initialize similarity engine
    sim_engine = SimilarityEngine(user_item_matrix)
    
    # Calculate similarities
    movie_sim = sim_engine.calculate_movie_similarity(method='cosine')
    user_sim = sim_engine.calculate_user_similarity(method='cosine')
    
    # Get similar movies
    similar_movies = sim_engine.get_similar_movies(movie_id=1, n_similar=5)
    print(f"\nMovies similar to Movie 1:")
    for movie_id, similarity, title in similar_movies:
        print(f"  {title}: {similarity:.4f}")
    
    # Get similar users
    similar_users = sim_engine.get_similar_users(user_id=1, n_similar=5)
    print(f"\nUsers similar to User 1:")
    for user_id, similarity in similar_users:
        print(f"  User {user_id}: {similarity:.4f}")

if __name__ == "__main__":
    main()
