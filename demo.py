"""
Quick Demo Script for Movie Recommendation System

This script provides a simple way to test the recommendation system
without going through the full pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_with_sample_data():
    """Run a quick demo with sample data."""
    print("🎬 Movie Recommendation System - Quick Demo")
    print("=" * 50)
    
    try:
        # Try to use the recommendation engine
        from recommendation_engine import MovieRecommendationEngine
        import time
        
        print("🔧 Initializing recommendation engine...")
        start_time = time.time()
        engine = MovieRecommendationEngine()
        engine.load_data()
        load_time = time.time() - start_time
        print(f"⏱️  Engine loaded in {load_time:.2f} seconds")
        
        # Test with a sample user
        test_user_id = 123
        print(f"\n🎯 Testing recommendations for User {test_user_id}")
        
        # Try different recommendation methods
        methods = [
            ("Similarity-based", lambda: engine.recommend_movies_similarity(test_user_id, 'item_based', 5)),
            ("ML-based", lambda: engine.recommend_movies_ml(test_user_id, 'XGBoost', 5)),
            ("Hybrid", lambda: engine.get_hybrid_recommendations(test_user_id, 5))
        ]
        
        for method_name, get_recs in methods:
            print(f"\n📋 {method_name} Recommendations:")
            try:
                start_time = time.time()
                recommendations = get_recs()
                rec_time = time.time() - start_time
                print(f"⏱️  Generated in {rec_time:.3f} seconds")
                
                if recommendations:
                    for i, rec in enumerate(recommendations, 1):
                        title = rec.get('title', f"Movie {rec.get('movieId', 'Unknown')}")
                        rating = rec.get('predicted_rating', rec.get('hybrid_score', 'N/A'))
                        print(f"   {i}. {title}")
                        if isinstance(rating, (int, float)):
                            print(f"      Rating: {rating:.2f} ⭐")
                else:
                    print("   No recommendations available")
            except Exception as e:
                print(f"   Error: {e}")
        
        # Test rating prediction
        print(f"\n🔮 Rating Prediction Test:")
        try:
            start_time = time.time()
            prediction = engine.predict_user_rating(test_user_id, 1)
            pred_time = time.time() - start_time
            print(f"⏱️  Prediction made in {pred_time:.3f} seconds")
            print(f"   User {test_user_id} would rate '{prediction['title']}': {prediction['predicted_rating']:.2f} ⭐")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test new user recommendations
        print(f"\n🆕 New User Recommendations Test:")
        try:
            # Test different methods for new users
            test_methods = [
                ('genre', ['Action', 'Comedy']),
                ('popularity_genre', ['Action', 'Comedy']),
                ('popularity', None)
            ]
            
            for method, genres in test_methods:
                print(f"\n  📋 Method: {method} with genres: {genres}")
                start_time = time.time()
                new_user_recs = engine.get_new_user_recommendations(
                    user_preferences=None,
                    preferred_genres=genres,
                    n_recommendations=3,
                    method=method
                )
                new_user_time = time.time() - start_time
                print(f"⏱️  Generated in {new_user_time:.3f} seconds")
                
                if new_user_recs:
                    for i, rec in enumerate(new_user_recs, 1):
                        title = rec.get('title', f"Movie {rec.get('movieId', 'Unknown')}")
                        rating = rec.get('predicted_rating', 'N/A')
                        reason = rec.get('reason', 'No reason provided')
                        print(f"     {i}. {title}")
                        if isinstance(rating, (int, float)):
                            print(f"        Rating: {rating:.2f} ⭐")
                        print(f"        Reason: {reason}")
                else:
                    print("     No recommendations available")
                        
        except Exception as e:
            print(f"   Error: {e}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Running basic demo instead...")
        basic_demo()
    except Exception as e:
        print(f"❌ Error: {e}")
        basic_demo()

def basic_demo():
    """Run a basic demo with hardcoded recommendations."""
    print("\n🎪 Basic Demo Mode")
    print("=" * 30)
    
    # Simulate some popular movie recommendations
    popular_movies = [
        {"title": "The Shawshank Redemption (1994)", "rating": 4.5, "genres": "Crime|Drama"},
        {"title": "The Godfather (1972)", "rating": 4.4, "genres": "Crime|Drama"},
        {"title": "Pulp Fiction (1994)", "rating": 4.3, "genres": "Crime|Drama|Thriller"},
        {"title": "The Dark Knight (2008)", "rating": 4.3, "genres": "Action|Crime|Drama"},
        {"title": "Inception (2010)", "rating": 4.2, "genres": "Action|Sci-Fi|Thriller"}
    ]
    
    print("🎬 Sample Popular Movie Recommendations:")
    for i, movie in enumerate(popular_movies, 1):
        print(f"   {i}. {movie['title']}")
        print(f"      Rating: {movie['rating']} ⭐ | Genres: {movie['genres']}")
    
    print(f"\n📝 Note: This is a basic demo. For full functionality:")
    print(f"   1. Download MovieLens 20M dataset to data/raw/")
    print(f"   2. Run: python main.py --action preprocess")
    print(f"   3. Run: python main.py --action train")
    print(f"   4. Run: python main.py --action recommend --user-id 123")

def interactive_demo():
    """Run an interactive demo."""
    print("\n🎮 Interactive Demo")
    print("=" * 20)
    
    while True:
        print("\nChoose an option:")
        print("1. Get movie recommendations")
        print("2. Predict movie rating")
        print("3. View system info")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            try:
                user_id = int(input("Enter User ID: "))
                print(f"\n🎯 Recommendations for User {user_id}:")
                # This would call the actual recommendation engine
                print("   [Demo] Showing sample recommendations...")
                basic_demo()
            except ValueError:
                print("❌ Please enter a valid user ID")
        
        elif choice == "2":
            try:
                user_id = int(input("Enter User ID: "))
                movie_id = int(input("Enter Movie ID: "))
                print(f"\n🔮 Predicted rating for User {user_id}, Movie {movie_id}: 4.2 ⭐")
            except ValueError:
                print("❌ Please enter valid IDs")
        
        elif choice == "3":
            print("\n📊 System Information:")
            print("   Status: Demo Mode")
            print("   Available Models: Linear Regression, Random Forest, XGBoost")
            print("   Features: 25+ engineered features")
            print("   Methods: Collaborative Filtering, ML-based, Hybrid")
        
        elif choice == "4":
            print("👋 Thanks for trying the Movie Recommendation System!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Movie Recommendation System Demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    parser.add_argument("--evaluate", action="store_true", help="Run quick evaluation")
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    elif args.evaluate:
        # Import and run quick evaluation
        try:
            from quick_eval import quick_evaluation
            quick_evaluation()
        except ImportError:
            print("❌ Evaluation module not found. Please ensure quick_eval.py exists.")
    else:
        demo_with_sample_data()

if __name__ == "__main__":
    main()
