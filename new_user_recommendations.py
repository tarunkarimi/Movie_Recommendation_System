"""
🆕 New User Recommendation Tool

Command-line interface for getting recommendations for new users
without requiring them to be in the existing dataset.
"""

import argparse
import sys
from pathlib import Path
import json

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from recommendation_engine import MovieRecommendationEngine
except ImportError as e:
    print(f"Error importing recommendation engine: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

def get_user_input_ratings(movies_df):
    """Interactive rating input for new users."""
    print("\n🎬 Rate some movies to get better recommendations")
    print("Enter ratings from 1-5 (0 to skip):")
    print("-" * 50)
    
    user_ratings = {}
    
    # Get some popular movies for rating
    if movies_df is not None:
        sample_movies = movies_df.head(20)  # First 20 movies as sample
        
        for _, movie in sample_movies.iterrows():
            movie_id = movie['movieId']
            title = movie['title']
            
            while True:
                try:
                    rating = input(f"🎬 {title}: ").strip()
                    if rating == '' or rating == '0':
                        break
                    
                    rating_val = float(rating)
                    if 1 <= rating_val <= 5:
                        user_ratings[movie_id] = rating_val
                        break
                    else:
                        print("   Please enter a rating between 1-5 (or 0 to skip)")
                except ValueError:
                    print("   Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    return user_ratings
    
    return user_ratings

def get_user_input_genres():
    """Interactive genre selection for new users."""
    available_genres = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    
    print("\n🎭 Select your favorite genres:")
    print("Available genres:")
    for i, genre in enumerate(available_genres, 1):
        print(f"  {i:2}. {genre}")
    
    print("\nEnter genre numbers separated by commas (e.g., 1,5,7):")
    
    while True:
        try:
            selection = input("Your selection: ").strip()
            if not selection:
                return []
            
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_genres = []
            
            for idx in indices:
                if 0 <= idx < len(available_genres):
                    selected_genres.append(available_genres[idx])
                else:
                    print(f"Invalid genre number: {idx + 1}")
                    continue
            
            return selected_genres
            
        except ValueError:
            print("Please enter valid numbers separated by commas")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return []

def display_recommendations(recommendations, method):
    """Display recommendations in a formatted way."""
    if not recommendations:
        print("❌ No recommendations available.")
        return
    
    print(f"\n🎯 Your Recommendations ({method}):")
    print("=" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        title = rec.get('title', f"Movie ID: {rec.get('movieId', 'Unknown')}")
        rating = rec.get('predicted_rating', 0)
        reason = rec.get('reason', 'Recommended for you')
        
        print(f"{i:2}. {title}")
        print(f"    ⭐ Predicted Rating: {rating:.1f}")
        print(f"    💡 {reason}")
        
        if 'genres' in rec or 'matching_genres' in rec:
            genres = rec.get('matching_genres', rec.get('genres', ''))
            if genres:
                if isinstance(genres, list):
                    genres_str = ', '.join(genres)
                else:
                    genres_str = str(genres)
                print(f"    🎭 Genres: {genres_str}")
        
        if 'rating_count' in rec:
            print(f"    📊 Based on {rec['rating_count']} ratings")
        
        print()

def save_user_profile(user_ratings, preferred_genres, recommendations, filename):
    """Save user profile and recommendations to file."""
    profile = {
        'user_ratings': user_ratings,
        'preferred_genres': preferred_genres,
        'recommendations': recommendations,
        'timestamp': str(pd.Timestamp.now())
    }
    
    with open(filename, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f"✅ Profile saved to {filename}")

def main():
    """Main function for new user recommendations."""
    parser = argparse.ArgumentParser(description="Get movie recommendations for new users")
    parser.add_argument('--method', choices=['popularity', 'genre', 'popularity_genre', 'similarity'], 
                       default='popularity_genre', help='Recommendation method')
    parser.add_argument('--n-recommendations', type=int, default=10, 
                       help='Number of recommendations to generate')
    parser.add_argument('--interactive', action='store_true', 
                       help='Interactive mode for rating movies and selecting genres')
    parser.add_argument('--genres', nargs='+', help='Preferred genres (space-separated)')
    parser.add_argument('--save-profile', help='Save user profile to file')
    parser.add_argument('--quick', action='store_true', help='Quick mode with popular recommendations only')
    
    args = parser.parse_args()
    
    print("🎬 Movie Recommendation System - New User Tool")
    print("=" * 50)
    
    # Initialize recommendation engine
    print("🔄 Loading recommendation system...")
    try:
        engine = MovieRecommendationEngine()
        engine.load_data()
        print("✅ System loaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not load full system ({e})")
        print("📝 Using fallback mode with synthetic data")
        engine = None
    
    # Quick mode - just get popular recommendations
    if args.quick:
        print("\n⚡ Quick Mode: Getting popular movie recommendations...")
        if engine:
            recommendations = engine.get_new_user_recommendations(
                n_recommendations=args.n_recommendations,
                method='popularity'
            )
        else:
            # Fallback popular movies
            popular_movies = [
                "The Shawshank Redemption", "The Godfather", "The Dark Knight",
                "Pulp Fiction", "Forrest Gump", "Inception", "The Matrix",
                "Goodfellas", "The Lord of the Rings", "Se7en"
            ]
            recommendations = []
            for i, title in enumerate(popular_movies[:args.n_recommendations]):
                recommendations.append({
                    'movieId': i + 1,
                    'title': title,
                    'predicted_rating': 4.5,
                    'reason': 'Highly rated popular movie'
                })
        
        display_recommendations(recommendations, "Popular Movies")
        return
    
    # Get user preferences
    user_ratings = {}
    preferred_genres = []
    
    if args.interactive:
        # Interactive mode
        movies_df = getattr(engine, 'movies_df', None) if engine else None
        user_ratings = get_user_input_ratings(movies_df)
        preferred_genres = get_user_input_genres()
    else:
        # Command line arguments
        if args.genres:
            preferred_genres = args.genres
    
    # Display user profile
    if user_ratings or preferred_genres:
        print("\n👤 Your Profile:")
        print("-" * 30)
        if user_ratings:
            print(f"🎬 Movies Rated: {len(user_ratings)}")
            avg_rating = sum(user_ratings.values()) / len(user_ratings)
            print(f"⭐ Average Rating: {avg_rating:.1f}")
        if preferred_genres:
            print(f"🎭 Preferred Genres: {', '.join(preferred_genres)}")
    
    # Generate recommendations
    print(f"\n🚀 Generating recommendations using method: {args.method}")
    
    try:
        if engine:
            recommendations = engine.get_new_user_recommendations(
                user_preferences=user_ratings if user_ratings else None,
                preferred_genres=preferred_genres if preferred_genres else None,
                n_recommendations=args.n_recommendations,
                method=args.method
            )
        else:
            # Fallback recommendations based on preferences
            recommendations = generate_fallback_recommendations(
                user_ratings, preferred_genres, args.n_recommendations
            )
        
        display_recommendations(recommendations, args.method.replace('_', ' ').title())
        
        # Save profile if requested
        if args.save_profile:
            save_user_profile(user_ratings, preferred_genres, recommendations, args.save_profile)
            
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        print("📝 This might be due to missing data or models.")

def generate_fallback_recommendations(user_ratings, preferred_genres, n_recommendations):
    """Generate simple fallback recommendations."""
    import random
    
    genre_movies = {
        'Action': ['The Dark Knight', 'Mad Max: Fury Road', 'John Wick'],
        'Comedy': ['The Hangover', 'Superbad', 'Anchorman'],
        'Drama': ['The Shawshank Redemption', 'Forrest Gump', 'The Godfather'],
        'Romance': ['Titanic', 'The Notebook', 'Casablanca'],
        'Sci-Fi': ['The Matrix', 'Inception', 'Blade Runner'],
        'Horror': ['The Shining', 'Halloween', 'The Exorcist']
    }
    
    recommendations = []
    used_movies = set()
    
    # Recommendations based on genres
    if preferred_genres:
        for genre in preferred_genres:
            if genre in genre_movies:
                for movie in genre_movies[genre]:
                    if movie not in used_movies and len(recommendations) < n_recommendations:
                        recommendations.append({
                            'movieId': len(recommendations) + 1,
                            'title': movie,
                            'predicted_rating': random.uniform(4.0, 5.0),
                            'reason': f'Popular {genre} movie'
                        })
                        used_movies.add(movie)
    
    # Fill with popular movies
    popular_movies = [
        'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
        'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix'
    ]
    
    for movie in popular_movies:
        if movie not in used_movies and len(recommendations) < n_recommendations:
            recommendations.append({
                'movieId': len(recommendations) + 1,
                'title': movie,
                'predicted_rating': random.uniform(4.2, 4.8),
                'reason': 'Highly rated popular movie'
            })
            used_movies.add(movie)
    
    return recommendations

if __name__ == "__main__":
    try:
        import pandas as pd
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your installation and try again.")
