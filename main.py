"""
Main Application Entry Point for Movie Recommendation System

This script provides a command-line interface to run the complete
movie recommendation pipeline.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--action", choices=["preprocess", "train", "recommend", "demo"], 
                       default="demo", help="Action to perform")
    parser.add_argument("--user-id", type=int, default=123, help="User ID for recommendations")
    parser.add_argument("--movie-id", type=int, help="Movie ID for rating prediction")
    parser.add_argument("--method", choices=["similarity", "ml", "hybrid"], 
                       default="hybrid", help="Recommendation method")
    parser.add_argument("--n-recs", type=int, default=10, help="Number of recommendations")
    
    args = parser.parse_args()
    
    print("🎬 Movie Recommendation System")
    print("=" * 50)
    
    if args.action == "preprocess":
        print("📊 Data Preprocessing")
        try:
            from data_preprocessing import DataPreprocessor
            
            preprocessor = DataPreprocessor()
            ratings_df, movies_df = preprocessor.load_data()
            
            validation_results = preprocessor.validate_data()
            print("Validation Results:", validation_results)
            
            processed_df = preprocessor.clean_data()
            stats = preprocessor.get_data_statistics()
            
            print("\nDataset Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
            
            user_item_matrix = preprocessor.create_user_item_matrix()
            preprocessor.save_processed_data()
            
            print("✅ Data preprocessing completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
    
    elif args.action == "train":
        print("🧠 Model Training")
        try:
            # First check if we have required data
            data_path = Path("data/processed/engineered_features.csv")
            if not data_path.exists():
                print("⚠️  Engineered features not found. Creating features first...")
                
                from data_preprocessing import DataPreprocessor
                from feature_engineering import FeatureEngineer
                
                # Load and preprocess data
                preprocessor = DataPreprocessor()
                ratings_df, movies_df = preprocessor.load_data()
                processed_df = preprocessor.clean_data()
                
                # Create features
                feature_engineer = FeatureEngineer(processed_df)
                features_df = feature_engineer.create_all_features(movies_df)
                feature_engineer.save_features()
                
                print("✅ Features created successfully!")
            
            # Train ML models
            from ml_models import MLModelTrainer
            import pandas as pd
            
            features_df = pd.read_csv("data/processed/engineered_features.csv")
            trainer = MLModelTrainer(features_df)
            
            print("Training models...")
            trainer.train_all_models()
            
            # Show results
            comparison = trainer.get_model_comparison()
            print("\n📈 Model Comparison:")
            print(comparison)
            
            # Save models
            trainer.save_models()
            
            print("✅ Model training completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in training: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.action == "recommend":
        print(f"🎯 Generating Recommendations for User {args.user_id}")
        try:
            from recommendation_engine import MovieRecommendationEngine
            
            engine = MovieRecommendationEngine()
            engine.load_data()
            
            if args.method == "similarity":
                recommendations = engine.recommend_movies_similarity(
                    args.user_id, 'item_based', args.n_recs
                )
            elif args.method == "ml":
                recommendations = engine.recommend_movies_ml(
                    args.user_id, 'XGBoost', args.n_recs
                )
            else:  # hybrid
                recommendations = engine.get_hybrid_recommendations(
                    args.user_id, args.n_recs
                )
            
            print(f"\n🍿 Top {len(recommendations)} Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                rating = rec.get('predicted_rating', rec.get('hybrid_score', 'N/A'))
                print(f"{i:2d}. {rec['title']}")
                print(f"     Rating: {rating:.2f} | Method: {rec.get('method', 'unknown')}")
                if 'genres' in rec:
                    print(f"     Genres: {rec['genres']}")
                print()
            
            # If movie ID provided, predict rating
            if args.movie_id:
                prediction = engine.predict_user_rating(args.user_id, args.movie_id)
                print(f"📊 Rating Prediction:")
                print(f"User {args.user_id} would rate '{prediction['title']}': {prediction['predicted_rating']:.2f}")
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
    
    else:  # demo
        print("🎪 Running Demo")
        try:
            from recommendation_engine import MovieRecommendationEngine
            
            engine = MovieRecommendationEngine()
            engine.load_data()
            
            test_user_id = args.user_id
            print(f"\n=== Demo for User {test_user_id} ===")
            
            # Try different methods
            methods = [
                ("Item-based Collaborative Filtering", "similarity"),
                ("Machine Learning (XGBoost)", "ml"),
                ("Hybrid Approach", "hybrid")
            ]
            
            for method_name, method_type in methods:
                print(f"\n🔹 {method_name}:")
                try:
                    if method_type == "similarity":
                        recs = engine.recommend_movies_similarity(test_user_id, 'item_based', 5)
                    elif method_type == "ml":
                        recs = engine.recommend_movies_ml(test_user_id, 'XGBoost', 5)
                    else:
                        recs = engine.get_hybrid_recommendations(test_user_id, 5)
                    
                    for i, rec in enumerate(recs, 1):
                        rating = rec.get('predicted_rating', rec.get('hybrid_score', 'N/A'))
                        print(f"   {i}. {rec['title']} - Rating: {rating:.2f}")
                
                except Exception as e:
                    print(f"   Error: {e}")
            
            # Demo rating prediction
            print(f"\n🎯 Rating Prediction Demo:")
            test_movie_id = 1
            prediction = engine.predict_user_rating(test_user_id, test_movie_id)
            print(f"   User {test_user_id} → '{prediction['title']}': {prediction['predicted_rating']:.2f}")
            
            print(f"\n✅ Demo completed!")
            print("\n📝 Next Steps:")
            print("   1. Download MovieLens 20M dataset to data/raw/")
            print("   2. Run: python main.py --action preprocess")
            print("   3. Run: python main.py --action train")
            print("   4. Run: python main.py --action recommend --user-id 123")
            
        except Exception as e:
            print(f"❌ Error in demo: {e}")
            import traceback
            traceback.print_exc()

def setup_data_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/raw",
        "data/processed", 
        "models",
        "notebooks"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    setup_data_directories()
    main()
