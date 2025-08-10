"""
Main Recommendation Engine

This module combines all components to provide a unified recommendation system
that uses both collaborative filtering and machine learning approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import pickle
import json

class MovieRecommendationEngine:
    """
    Main recommendation engine that combines collaborative filtering and ML models.
    """
    
    def __init__(self, data_path: str = "data"):
        """
        Initialize the recommendation engine.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        
        # Data storage
        self.user_item_matrix = None
        self.movies_df = None
        self.ratings_df = None
        
        # Similarity matrices
        self.movie_similarity_matrix = None
        self.user_similarity_matrix = None
        
        # ML models
        self.ml_models = {}
        self.feature_scaler = None
        self.feature_columns = []
        
        # Statistics
        self.global_mean = 3.5
        self.user_means = {}
        self.movie_means = {}
        
        # Caching for performance
        self._movie_info_cache = {}
        self._popular_movies_cache = None
        self._genre_movies_cache = {}
        
        print("Movie Recommendation Engine initialized")
    
    def load_data(self):
        """Load all necessary data and models."""
        print("Loading data and models...")
        self.ratings_df = pd.read_csv("data/raw/ratings.csv")
        self.movies_df = pd.read_csv("data/raw/movies.csv")
        print(f"✅ Ratings loaded: {self.ratings_df.shape}")
        print(f"✅ Movies loaded: {self.movies_df.shape}")
        
        try:
            # Load processed data
            processed_path = self.data_path / "processed"
            
            if (processed_path / "processed_ratings.csv").exists():
                self.ratings_df = pd.read_csv(processed_path / "processed_ratings.csv")
                print(f"Loaded {len(self.ratings_df):,} ratings")
            
            # Load movies data
            raw_path = self.data_path / "raw"
            if (raw_path / "movies.csv").exists():
                self.movies_df = pd.read_csv(raw_path / "movies.csv")
                print(f"Loaded {len(self.movies_df):,} movies")
            
            # Load similarity matrices
            models_path = Path("models")
            
            if (models_path / "movie_similarity_matrix.pkl").exists():
                with open(models_path / "movie_similarity_matrix.pkl", "rb") as f:
                    self.movie_similarity_matrix = pickle.load(f)
                print("Loaded movie similarity matrix")
            
            if (models_path / "user_similarity_matrix.pkl").exists():
                with open(models_path / "user_similarity_matrix.pkl", "rb") as f:
                    self.user_similarity_matrix = pickle.load(f)
                print("Loaded user similarity matrix")
            
            # Load ML models
            if (models_path / "model_metadata.json").exists():
                with open(models_path / "model_metadata.json", "r") as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', [])
            
            # Load feature scaler
            if (models_path / "feature_scaler.pkl").exists():
                with open(models_path / "feature_scaler.pkl", "rb") as f:
                    self.feature_scaler = pickle.load(f)
                print("Loaded feature scaler")
            
            # Load individual models
            model_files = {
                'XGBoost': 'xgboost_model.pkl',
                'Random Forest': 'random_forest_model.pkl',
                'LightGBM': 'lightgbm_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self.ml_models[model_name] = pickle.load(f)
                    print(f"Loaded {model_name} model")
            
            # Calculate basic statistics
            self._calculate_statistics()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Some components may not be available")
    
    def _calculate_statistics(self):
        """Calculate basic statistics from the data."""
        if self.ratings_df is not None:
            self.global_mean = self.ratings_df['rating'].mean()
            
            # User means
            user_stats = self.ratings_df.groupby('userId')['rating'].mean()
            self.user_means = user_stats.to_dict()
            
            # Movie means
            movie_stats = self.ratings_df.groupby('movieId')['rating'].mean()
            self.movie_means = movie_stats.to_dict()
            
            print(f"Calculated statistics - Global mean: {self.global_mean:.2f}")
    
    def get_movie_info(self, movie_id: int) -> Dict[str, any]:
        """Get movie information by ID with caching."""
        # Check cache first
        if movie_id in self._movie_info_cache:
            return self._movie_info_cache[movie_id].copy()
        
        if self.movies_df is None:
            movie_info = {'movieId': movie_id, 'title': f'Movie_{movie_id}', 'genres': 'Unknown'}
        else:
            movie_data = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_data) == 0:
                movie_info = {'movieId': movie_id, 'title': f'Movie_{movie_id}', 'genres': 'Unknown'}
            else:
                movie_info = movie_data.iloc[0].to_dict()
        
        # Cache the result
        self._movie_info_cache[movie_id] = movie_info.copy()
        return movie_info
    
    def get_new_user_recommendations(
        self, 
        user_preferences: Dict[int, float] = None,
        preferred_genres: List[str] = None,
        n_recommendations: int = 10,
        method: str = "popularity_genre"
    ) -> List[Dict]:
        """
        Get recommendations for a completely new user.
        
        Args:
            user_preferences: Dict of {movie_id: rating} for movies the user has rated
            preferred_genres: List of preferred genres
            n_recommendations: Number of recommendations to return
            method: Recommendation method ('popularity', 'genre', 'popularity_genre', 'similarity')
            
        Returns:
            List of movie recommendations with explanations
        """
        print(f"Generating recommendations for new user using method: {method}")
        
        if method == "popularity":
            return self._get_popular_movie_recommendations(n_recommendations)
        elif method == "genre" and preferred_genres:
            return self._get_genre_based_recommendations(preferred_genres, n_recommendations)
        elif method == "popularity_genre" and preferred_genres:
            return self._get_popularity_genre_recommendations(preferred_genres, n_recommendations)
        elif method == "similarity" and user_preferences:
            return self._get_similarity_based_new_user_recommendations(user_preferences, n_recommendations)
        else:
            # Default: most popular movies
            return self._get_popular_movie_recommendations(n_recommendations)
    
    def _get_popular_movie_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Get most popular movies based on rating count and average rating with caching."""
        if self.ratings_df is None:
            return self._get_fallback_recommendations(n_recommendations)
        
        # Check cache first
        if self._popular_movies_cache is not None and len(self._popular_movies_cache) >= n_recommendations:
            return self._popular_movies_cache[:n_recommendations]
        
        try:
            # Calculate movie popularity metrics
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['count', 'mean']
            }).round(2)
            movie_stats.columns = ['rating_count', 'avg_rating']
            
            # Filter movies with minimum ratings
            min_ratings = max(50, len(self.ratings_df) // 1000)  # Adaptive threshold
            popular_movies = movie_stats[movie_stats['rating_count'] >= min_ratings].copy()
            
            # Calculate popularity score (weighted average rating and count)
            if len(popular_movies) > 0:
                max_count = popular_movies['rating_count'].max()
                popular_movies.loc[:, 'popularity_score'] = (
                    popular_movies['avg_rating'] * 0.7 + 
                    (popular_movies['rating_count'] / max_count) * 5 * 0.3
                )
            else:
                # Fallback if no movies meet the criteria
                return self._get_fallback_recommendations(n_recommendations)
            
            # Sort by popularity score
            top_movies = popular_movies.nlargest(50, 'popularity_score')  # Cache more for future use
            
            recommendations = []
            for movie_id, stats in top_movies.iterrows():
                movie_info = self.get_movie_info(movie_id)
                movie_info.update({
                    'predicted_rating': stats['avg_rating'],
                    'popularity_score': stats['popularity_score'],
                    'rating_count': int(stats['rating_count']),
                    'reason': f"Popular movie with {int(stats['rating_count'])} ratings (avg: {stats['avg_rating']:.1f}⭐)"
                })
                recommendations.append(movie_info)
            
            # Cache the results
            self._popular_movies_cache = recommendations
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error in popular recommendations: {e}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Generate fallback recommendations when other methods fail."""
        fallback_movies = [
            {"title": "The Shawshank Redemption", "rating": 4.5, "reason": "Highly rated classic"},
            {"title": "The Godfather", "rating": 4.4, "reason": "Legendary crime drama"},
            {"title": "The Dark Knight", "rating": 4.4, "reason": "Acclaimed superhero film"},
            {"title": "Pulp Fiction", "rating": 4.3, "reason": "Iconic Tarantino masterpiece"},
            {"title": "Forrest Gump", "rating": 4.2, "reason": "Beloved feel-good movie"},
            {"title": "Inception", "rating": 4.2, "reason": "Mind-bending sci-fi thriller"},
            {"title": "The Matrix", "rating": 4.1, "reason": "Revolutionary sci-fi action"},
            {"title": "Goodfellas", "rating": 4.1, "reason": "Classic crime saga"},
            {"title": "The Lord of the Rings", "rating": 4.0, "reason": "Epic fantasy adventure"},
            {"title": "Se7en", "rating": 4.0, "reason": "Gripping psychological thriller"}
        ]
        
        recommendations = []
        for i, movie in enumerate(fallback_movies[:n_recommendations]):
            recommendations.append({
                'movieId': 1000 + i,
                'title': movie['title'],
                'predicted_rating': movie['rating'],
                'reason': movie['reason']
            })
        
        return recommendations
    
    def _get_genre_based_recommendations(self, preferred_genres: List[str], n_recommendations: int) -> List[Dict]:
        """Get recommendations based on preferred genres."""
        if self.movies_df is None or 'genres' not in self.movies_df.columns:
            return self._get_popular_movie_recommendations(n_recommendations)
        
        try:
            # Filter movies by preferred genres
            genre_movies = []
            for _, movie in self.movies_df.iterrows():
                if pd.notna(movie.get('genres', '')) and movie['genres'] != '(no genres listed)':
                    movie_genres = movie['genres'].split('|')
                    if any(genre in movie_genres for genre in preferred_genres):
                        genre_movies.append(movie['movieId'])
            
            if not genre_movies:
                return self._get_popular_movie_recommendations(n_recommendations)
            
            # Get ratings for genre movies
            if self.ratings_df is not None:
                genre_ratings = self.ratings_df[self.ratings_df['movieId'].isin(genre_movies)]
                
                if len(genre_ratings) > 0:
                    # Calculate stats for genre movies
                    movie_stats = genre_ratings.groupby('movieId').agg({
                        'rating': ['count', 'mean']
                    }).round(2)
                    movie_stats.columns = ['rating_count', 'avg_rating']
                    
                    # Filter by minimum ratings
                    min_ratings = max(10, len(genre_ratings) // 100)
                    good_movies = movie_stats[
                        (movie_stats['rating_count'] >= min_ratings) & 
                        (movie_stats['avg_rating'] >= 3.5)
                    ]
                    
                    if len(good_movies) >= n_recommendations:
                        top_movies = good_movies.nlargest(n_recommendations, 'avg_rating')
                    else:
                        top_movies = good_movies.nlargest(min(len(good_movies), n_recommendations), 'avg_rating')
                        # Fill remaining with popular genre movies
                        remaining = n_recommendations - len(top_movies)
                        if remaining > 0:
                            all_genre_movies = movie_stats.nlargest(remaining, 'rating_count')
                            top_movies = pd.concat([top_movies, all_genre_movies]).drop_duplicates()
                    
                    recommendations = []
                    for movie_id, stats in top_movies.iterrows():
                        movie_info = self.get_movie_info(movie_id)
                        movie_genres = self.movies_df[self.movies_df['movieId'] == movie_id]['genres'].iloc[0]
                        matching_genres = [g for g in preferred_genres if g in movie_genres]
                        
                        movie_info.update({
                            'predicted_rating': stats['avg_rating'],
                            'rating_count': int(stats['rating_count']),
                            'matching_genres': matching_genres,
                            'reason': f"Matches your preferred genres: {', '.join(matching_genres)}"
                        })
                        recommendations.append(movie_info)
                    
                    return recommendations
            
            # Fallback: just return random genre movies
            selected_movies = np.random.choice(genre_movies, min(n_recommendations, len(genre_movies)), replace=False)
            recommendations = []
            for movie_id in selected_movies:
                movie_info = self.get_movie_info(movie_id)
                movie_info['reason'] = f"Matches your preferred genres: {', '.join(preferred_genres)}"
                recommendations.append(movie_info)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in genre recommendations: {e}")
            return self._get_popular_movie_recommendations(n_recommendations)
    
    def _get_popularity_genre_recommendations(self, preferred_genres: List[str], n_recommendations: int) -> List[Dict]:
        """Combine popularity and genre preferences."""
        try:
            # Get genre-based recommendations
            genre_recs = self._get_genre_based_recommendations(preferred_genres, n_recommendations * 2)
            
            # Get popular recommendations
            popular_recs = self._get_popular_movie_recommendations(n_recommendations)
            
            # Combine and remove duplicates
            combined_movies = {}
            
            # Add genre recommendations with higher weight
            for rec in genre_recs:
                movie_id = rec['movieId']
                score = rec.get('predicted_rating', 3.5) * 1.2  # Boost genre matches
                combined_movies[movie_id] = {**rec, 'combined_score': score}
            
            # Add popular recommendations
            for rec in popular_recs:
                movie_id = rec['movieId']
                if movie_id not in combined_movies:
                    score = rec.get('predicted_rating', 3.5)
                    combined_movies[movie_id] = {**rec, 'combined_score': score}
                else:
                    # Boost score if it's both popular and matches genre
                    combined_movies[movie_id]['combined_score'] += 0.5
                    combined_movies[movie_id]['reason'] = "Popular movie that matches your preferred genres"
            
            # Sort by combined score and return top recommendations
            sorted_recs = sorted(combined_movies.values(), key=lambda x: x['combined_score'], reverse=True)
            return sorted_recs[:n_recommendations]
            
        except Exception as e:
            print(f"Error in popularity-genre recommendations: {e}")
            return self._get_popular_movie_recommendations(n_recommendations)
    
    def _get_similarity_based_new_user_recommendations(
        self, 
        user_preferences: Dict[int, float], 
        n_recommendations: int
    ) -> List[Dict]:
        """Get recommendations based on user's initial ratings using similarity."""
        if not user_preferences or self.ratings_df is None:
            return self._get_popular_movie_recommendations(n_recommendations)
        
        try:
            # Find similar users based on initial preferences
            similar_users = []
            
            for user_id in self.ratings_df['userId'].unique():
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                user_movies = set(user_ratings['movieId'].tolist())
                
                # Calculate overlap with new user's preferences
                common_movies = set(user_preferences.keys()) & user_movies
                
                if len(common_movies) >= 2:  # Need at least 2 movies in common
                    # Calculate similarity (simplified Pearson correlation)
                    new_user_ratings = [user_preferences[mid] for mid in common_movies]
                    existing_user_ratings = []
                    
                    for mid in common_movies:
                        rating = user_ratings[user_ratings['movieId'] == mid]['rating'].iloc[0]
                        existing_user_ratings.append(rating)
                    
                    if len(new_user_ratings) > 1:
                        correlation = np.corrcoef(new_user_ratings, existing_user_ratings)[0, 1]
                        if not np.isnan(correlation) and correlation > 0.3:
                            similar_users.append((user_id, correlation, len(common_movies)))
            
            if not similar_users:
                # No similar users found, fall back to popular recommendations
                return self._get_popular_movie_recommendations(n_recommendations)
            
            # Sort by similarity and number of common movies
            similar_users.sort(key=lambda x: (x[1], x[2]), reverse=True)
            top_similar_users = similar_users[:20]  # Use top 20 similar users
            
            # Get recommendations from similar users
            recommended_movies = {}
            
            for user_id, similarity, _ in top_similar_users:
                user_ratings = self.ratings_df[
                    (self.ratings_df['userId'] == user_id) & 
                    (~self.ratings_df['movieId'].isin(user_preferences.keys()))
                ]
                
                # Only consider highly rated movies
                good_ratings = user_ratings[user_ratings['rating'] >= 4.0]
                
                for _, rating_row in good_ratings.iterrows():
                    movie_id = rating_row['movieId']
                    rating = rating_row['rating']
                    
                    if movie_id not in recommended_movies:
                        recommended_movies[movie_id] = []
                    
                    weighted_rating = rating * similarity
                    recommended_movies[movie_id].append(weighted_rating)
            
            # Calculate average weighted ratings
            movie_scores = {}
            for movie_id, ratings in recommended_movies.items():
                if len(ratings) >= 2:  # Movie recommended by at least 2 similar users
                    movie_scores[movie_id] = np.mean(ratings)
            
            # Sort by score and get top recommendations
            sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
            
            recommendations = []
            for movie_id, score in sorted_movies[:n_recommendations]:
                movie_info = self.get_movie_info(movie_id)
                movie_info.update({
                    'predicted_rating': min(5.0, max(0.5, score)),
                    'similarity_score': score,
                    'reason': f"Recommended by users with similar taste (score: {score:.2f})"
                })
                recommendations.append(movie_info)
            
            # Fill with popular recommendations if needed
            if len(recommendations) < n_recommendations:
                remaining = n_recommendations - len(recommendations)
                popular_recs = self._get_popular_movie_recommendations(remaining * 2)
                
                # Add popular movies not already recommended
                recommended_ids = {rec['movieId'] for rec in recommendations}
                for rec in popular_recs:
                    if rec['movieId'] not in recommended_ids and len(recommendations) < n_recommendations:
                        rec['reason'] = "Popular movie (filled recommendation)"
                        recommendations.append(rec)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            print(f"Error in similarity-based new user recommendations: {e}")
            return self._get_popular_movie_recommendations(n_recommendations)
    
    def create_new_user_profile(
        self, 
        initial_ratings: Dict[int, float],
        preferred_genres: List[str] = None
    ) -> Dict:
        """
        Create a profile for a new user and get initial recommendations.
        
        Args:
            initial_ratings: Dict of {movie_id: rating} for initial ratings
            preferred_genres: List of preferred genres
            
        Returns:
            Dict containing user profile and recommendations
        """
        print(f"Creating profile for new user with {len(initial_ratings)} initial ratings")
        
        # Analyze user's initial ratings
        if initial_ratings:
            avg_rating = np.mean(list(initial_ratings.values()))
            rating_variance = np.var(list(initial_ratings.values()))
            
            # Determine user's rating style
            if rating_variance < 0.5:
                rating_style = "Consistent rater"
            elif rating_variance > 2.0:
                rating_style = "Diverse taste"
            else:
                rating_style = "Balanced rater"
            
            # Determine user's overall sentiment
            if avg_rating >= 4.0:
                sentiment = "Generous rater"
            elif avg_rating <= 3.0:
                sentiment = "Critical rater"
            else:
                sentiment = "Moderate rater"
        else:
            avg_rating = 3.5
            rating_style = "Unknown"
            sentiment = "Unknown"
        
        # Infer genres from initial ratings if not provided
        inferred_genres = []
        if initial_ratings and self.movies_df is not None:
            genre_ratings = {}
            for movie_id, rating in initial_ratings.items():
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
                if not movie_info.empty and 'genres' in movie_info.columns:
                    genres_str = movie_info.iloc[0]['genres']
                    if pd.notna(genres_str) and genres_str != '(no genres listed)':
                        for genre in genres_str.split('|'):
                            if genre not in genre_ratings:
                                genre_ratings[genre] = []
                            genre_ratings[genre].append(rating)
            
            # Find genres with high average ratings
            for genre, ratings in genre_ratings.items():
                if len(ratings) >= 1 and np.mean(ratings) >= 4.0:
                    inferred_genres.append(genre)
        
        # Use provided genres or inferred ones
        final_genres = preferred_genres if preferred_genres else inferred_genres[:5]
        
        # Get different types of recommendations
        recommendations = {}
        
        if initial_ratings:
            recommendations['similarity_based'] = self.get_new_user_recommendations(
                user_preferences=initial_ratings,
                n_recommendations=10,
                method="similarity"
            )
        
        if final_genres:
            recommendations['genre_based'] = self.get_new_user_recommendations(
                preferred_genres=final_genres,
                n_recommendations=10,
                method="popularity_genre"
            )
        
        recommendations['popular'] = self.get_new_user_recommendations(
            n_recommendations=10,
            method="popularity"
        )
        
        # Create user profile
        user_profile = {
            'initial_ratings': initial_ratings,
            'preferred_genres': final_genres,
            'inferred_genres': inferred_genres,
            'average_rating': avg_rating,
            'rating_style': rating_style,
            'sentiment': sentiment,
            'recommendations': recommendations,
            'profile_created': pd.Timestamp.now().isoformat()
        }
        
        return user_profile.iloc[0].to_dict()
    
    def recommend_movies_similarity(self, user_id: int, method: str = 'item_based', 
                                  n_recommendations: int = 10) -> List[Dict[str, any]]:
        """
        Recommend movies using similarity-based collaborative filtering.
        
        Args:
            user_id: Target user ID
            method: 'item_based' or 'user_based'
            n_recommendations: Number of recommendations
            
        Returns:
            List of movie recommendations with metadata
        """
        print(f"Generating {method} similarity recommendations for user {user_id}")
        
        if method == 'item_based' and self.movie_similarity_matrix is None:
            print("Movie similarity matrix not available")
            return self._get_popular_movies(n_recommendations)
        
        if method == 'user_based' and self.user_similarity_matrix is None:
            print("User similarity matrix not available")
            return self._get_popular_movies(n_recommendations)
        
        # This is a simplified implementation
        # In practice, you'd use the collaborative filtering module
        recommendations = []
        
        if method == 'item_based':
            # Get user's rated movies
            if self.ratings_df is not None:
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                
                if len(user_ratings) == 0:
                    return self._get_popular_movies(n_recommendations)
                
                # For each rated movie, find similar movies
                similar_movies = {}
                
                for _, rating_row in user_ratings.iterrows():
                    movie_id = rating_row['movieId']
                    user_rating = rating_row['rating']
                    
                    if movie_id in self.movie_similarity_matrix.index:
                        similarities = self.movie_similarity_matrix.loc[movie_id]
                        
                        # Get top similar movies
                        top_similar = similarities.nlargest(20)
                        
                        for sim_movie_id, similarity in top_similar.items():
                            if sim_movie_id != movie_id and similarity > 0.1:
                                if sim_movie_id not in similar_movies:
                                    similar_movies[sim_movie_id] = 0
                                similar_movies[sim_movie_id] += similarity * user_rating
                
                # Sort by score and get top recommendations
                sorted_movies = sorted(similar_movies.items(), key=lambda x: x[1], reverse=True)
                
                for movie_id, score in sorted_movies[:n_recommendations]:
                    movie_info = self.get_movie_info(movie_id)
                    movie_info['predicted_rating'] = min(5.0, max(0.5, score / 5))
                    movie_info['recommendation_score'] = score
                    movie_info['method'] = 'item_based_similarity'
                    recommendations.append(movie_info)
        
        return recommendations if recommendations else self._get_popular_movies(n_recommendations)
    
    def recommend_movies_ml(self, user_id: int, model_name: str = 'XGBoost', 
                           n_recommendations: int = 10) -> List[Dict[str, any]]:
        """
        Recommend movies using machine learning models.
        
        Args:
            user_id: Target user ID
            model_name: Name of the ML model to use
            n_recommendations: Number of recommendations
            
        Returns:
            List of movie recommendations with metadata
        """
        print(f"Generating ML recommendations for user {user_id} using {model_name}")
        
        if model_name not in self.ml_models:
            print(f"Model {model_name} not available")
            return self._get_popular_movies(n_recommendations)
        
        model = self.ml_models[model_name]
        recommendations = []
        
        # Get user statistics
        user_mean = self.user_means.get(user_id, self.global_mean)
        
        # Get user's rated movies to exclude them
        rated_movies = set()
        if self.ratings_df is not None:
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            rated_movies = set(user_ratings['movieId'].tolist())
        
        # Generate predictions for unrated movies
        movie_scores = []
        
        # Sample from available movies (in practice, you'd predict for all movies)
        if self.movies_df is not None:
            available_movies = self.movies_df['movieId'].tolist()
        else:
            available_movies = list(range(1, 1000))  # Dummy range
        
        for movie_id in available_movies[:500]:  # Limit for performance
            if movie_id not in rated_movies:
                movie_mean = self.movie_means.get(movie_id, self.global_mean)
                
                # Create feature vector (simplified)
                features = {
                    'user_mean': user_mean,
                    'movie_mean': movie_mean,
                    'rating_gmean': self.global_mean,
                    'user_bias': user_mean - self.global_mean,
                    'movie_bias': movie_mean - self.global_mean,
                }
                
                # Add other features with defaults
                for feature in self.feature_columns:
                    if feature not in features:
                        features[feature] = 0.0
                
                try:
                    predicted_rating = self._predict_with_ml_model(model, features, model_name)
                    movie_scores.append((movie_id, predicted_rating))
                except Exception as e:
                    print(f"Error predicting for movie {movie_id}: {e}")
                    continue
        
        # Sort by predicted rating and get top recommendations
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        for movie_id, predicted_rating in movie_scores[:n_recommendations]:
            movie_info = self.get_movie_info(movie_id)
            movie_info['predicted_rating'] = predicted_rating
            movie_info['method'] = f'ml_{model_name.lower()}'
            recommendations.append(movie_info)
        
        return recommendations if recommendations else self._get_popular_movies(n_recommendations)
    
    def _predict_with_ml_model(self, model, features: Dict[str, float], model_name: str) -> float:
        """Make prediction using ML model."""
        # Create feature vector
        feature_vector = np.zeros(len(self.feature_columns))
        for i, feature in enumerate(self.feature_columns):
            if feature in features:
                feature_vector[i] = features[feature]
        
        # Scale if needed (for linear models)
        if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            if self.feature_scaler is not None:
                feature_vector = self.feature_scaler.transform(feature_vector.reshape(1, -1))
                prediction = model.predict(feature_vector)[0]
            else:
                prediction = model.predict(feature_vector.reshape(1, -1))[0]
        else:
            prediction = model.predict(feature_vector.reshape(1, -1))[0]
        
        # Clamp to valid rating range
        return max(0.5, min(5.0, prediction))
    
    def _get_popular_movies(self, n_recommendations: int) -> List[Dict[str, any]]:
        """Get popular movies as fallback recommendations."""
        print("Using popular movies as fallback recommendations")
        
        recommendations = []
        
        if self.ratings_df is not None:
            # Get most rated movies
            movie_counts = self.ratings_df['movieId'].value_counts()
            popular_movies = movie_counts.head(n_recommendations)
            
            for movie_id, count in popular_movies.items():
                movie_info = self.get_movie_info(movie_id)
                movie_info['predicted_rating'] = self.movie_means.get(movie_id, self.global_mean)
                movie_info['popularity_score'] = count
                movie_info['method'] = 'popularity'
                recommendations.append(movie_info)
        else:
            # Dummy popular movies
            for i in range(1, n_recommendations + 1):
                movie_info = {
                    'movieId': i,
                    'title': f'Popular Movie {i}',
                    'genres': 'Unknown',
                    'predicted_rating': 4.0,
                    'method': 'popularity_fallback'
                }
                recommendations.append(movie_info)
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 10,
                                 weights: Dict[str, float] = None) -> List[Dict[str, any]]:
        """
        Get hybrid recommendations combining multiple methods.
        
        Args:
            user_id: Target user ID
            n_recommendations: Number of recommendations
            weights: Weights for different methods
            
        Returns:
            List of hybrid recommendations
        """
        if weights is None:
            weights = {
                'item_based': 0.3,
                'user_based': 0.2,
                'ml_xgboost': 0.4,
                'popularity': 0.1
            }
        
        print(f"Generating hybrid recommendations for user {user_id}")
        
        all_recommendations = {}
        
        # Get recommendations from different methods
        methods = [
            ('item_based', lambda: self.recommend_movies_similarity(user_id, 'item_based', n_recommendations * 2)),
            ('ml_xgboost', lambda: self.recommend_movies_ml(user_id, 'XGBoost', n_recommendations * 2)),
            ('popularity', lambda: self._get_popular_movies(n_recommendations))
        ]
        
        for method_name, get_recs_func in methods:
            if method_name in weights and weights[method_name] > 0:
                try:
                    recs = get_recs_func()
                    for rec in recs:
                        movie_id = rec['movieId']
                        if movie_id not in all_recommendations:
                            all_recommendations[movie_id] = {
                                'movie_info': rec,
                                'scores': {},
                                'total_score': 0
                            }
                        
                        score = rec.get('predicted_rating', 3.5) * weights[method_name]
                        all_recommendations[movie_id]['scores'][method_name] = score
                        all_recommendations[movie_id]['total_score'] += score
                except Exception as e:
                    print(f"Error getting {method_name} recommendations: {e}")
        
        # Sort by total score
        sorted_recs = sorted(all_recommendations.items(), 
                           key=lambda x: x[1]['total_score'], reverse=True)
        
        # Prepare final recommendations
        final_recommendations = []
        for movie_id, rec_data in sorted_recs[:n_recommendations]:
            movie_info = rec_data['movie_info'].copy()
            movie_info['hybrid_score'] = rec_data['total_score']
            movie_info['method_scores'] = rec_data['scores']
            movie_info['method'] = 'hybrid'
            final_recommendations.append(movie_info)
        
        return final_recommendations
    
    def predict_user_rating(self, user_id: int, movie_id: int, 
                           method: str = 'ml') -> Dict[str, any]:
        """
        Predict what rating a user would give to a specific movie.
        
        Args:
            user_id: Target user ID
            movie_id: Target movie ID
            method: Prediction method ('ml', 'item_based', 'user_based')
            
        Returns:
            Dictionary with prediction details
        """
        result = {
            'user_id': user_id,
            'movie_id': movie_id,
            'method': method,
            'predicted_rating': self.global_mean,
            'confidence': 0.5
        }
        
        if method == 'ml' and 'XGBoost' in self.ml_models:
            user_mean = self.user_means.get(user_id, self.global_mean)
            movie_mean = self.movie_means.get(movie_id, self.global_mean)
            
            features = {
                'user_mean': user_mean,
                'movie_mean': movie_mean,
                'rating_gmean': self.global_mean,
                'user_bias': user_mean - self.global_mean,
                'movie_bias': movie_mean - self.global_mean,
            }
            
            # Add other features with defaults
            for feature in self.feature_columns:
                if feature not in features:
                    features[feature] = 0.0
            
            try:
                predicted_rating = self._predict_with_ml_model(
                    self.ml_models['XGBoost'], features, 'XGBoost'
                )
                result['predicted_rating'] = predicted_rating
                result['confidence'] = 0.8
            except Exception as e:
                print(f"Error in ML prediction: {e}")
        
        # Add movie information
        movie_info = self.get_movie_info(movie_id)
        result.update(movie_info)
        
        return result

def main():
    """
    Example usage of the MovieRecommendationEngine.
    """
    print("Movie Recommendation Engine Demo")
    
    # Initialize engine
    engine = MovieRecommendationEngine()
    
    # Try to load data (will handle missing files gracefully)
    engine.load_data()
    
    # Test user ID
    test_user_id = 123
    
    print(f"\n=== Recommendations for User {test_user_id} ===")
    
    # Try different recommendation methods
    try:
        # Similarity-based recommendations
        print("\n1. Item-based Collaborative Filtering:")
        item_recs = engine.recommend_movies_similarity(test_user_id, 'item_based', 5)
        for i, rec in enumerate(item_recs, 1):
            print(f"   {i}. {rec['title']} - Rating: {rec.get('predicted_rating', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    try:
        # ML-based recommendations
        print("\n2. Machine Learning (XGBoost):")
        ml_recs = engine.recommend_movies_ml(test_user_id, 'XGBoost', 5)
        for i, rec in enumerate(ml_recs, 1):
            print(f"   {i}. {rec['title']} - Rating: {rec.get('predicted_rating', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    try:
        # Hybrid recommendations
        print("\n3. Hybrid Recommendations:")
        hybrid_recs = engine.get_hybrid_recommendations(test_user_id, 5)
        for i, rec in enumerate(hybrid_recs, 1):
            print(f"   {i}. {rec['title']} - Score: {rec.get('hybrid_score', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test rating prediction
    print(f"\n=== Rating Prediction ===")
    test_movie_id = 1
    prediction = engine.predict_user_rating(test_user_id, test_movie_id)
    print(f"User {test_user_id} would rate '{prediction['title']}': {prediction['predicted_rating']:.2f}")
    
    print("\n=== Demo Complete ===")
    print("Note: Full functionality requires processed data and trained models.")

if __name__ == "__main__":
    main()
