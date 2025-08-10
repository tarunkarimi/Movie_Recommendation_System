"""
Quick Evaluation Script for Movie Recommendation System

This script provides a fast evaluation of key metrics for the recommendation system.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def quick_evaluation():
    """Run a quick evaluation of the recommendation system."""
    print("🎬 Movie Recommendation System - Quick Evaluation")
    print("=" * 50)
    
    try:
        from recommendation_engine import MovieRecommendationEngine
        
        print("🔧 Loading recommendation engine...")
        start_time = time.time()
        engine = MovieRecommendationEngine()
        engine.load_data()
        load_time = time.time() - start_time
        print(f"✅ Engine loaded in {load_time:.2f} seconds")
        
        # Test users
        test_users = [1, 123, 456, 789, 1000]
        
        print(f"\n📊 Testing with {len(test_users)} sample users...")
        
        # 1. Response Time Evaluation
        print(f"\n⏱️  RESPONSE TIME EVALUATION")
        print("-" * 30)
        
        methods = {
            "Similarity": lambda uid: engine.recommend_movies_similarity(uid, 'item_based', 5),
            "ML": lambda uid: engine.recommend_movies_ml(uid, 'XGBoost', 5),
            "Hybrid": lambda uid: engine.get_hybrid_recommendations(uid, 5)
        }
        
        for method_name, method_func in methods.items():
            times = []
            success_count = 0
            
            for user_id in test_users:
                try:
                    start = time.time()
                    recommendations = method_func(user_id)
                    end = time.time()
                    
                    if recommendations and len(recommendations) > 0:
                        times.append(end - start)
                        success_count += 1
                except Exception:
                    continue
            
            if times:
                avg_time = np.mean(times)
                success_rate = success_count / len(test_users)
                print(f"  {method_name:12}: {avg_time:.3f}s avg, {success_rate:.0%} success")
            else:
                print(f"  {method_name:12}: Failed")
        
        # 2. Rating Prediction Accuracy
        print(f"\n📈 RATING PREDICTION ACCURACY")
        print("-" * 30)
        
        # Sample some ratings for testing
        sample_ratings = engine.ratings_df.sample(min(100, len(engine.ratings_df)), random_state=42)
        
        predictions = []
        actuals = []
        
        for _, row in sample_ratings.iterrows():
            try:
                user_id = row['userId']
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                # Use direct prediction method
                pred_result = engine.predict_user_rating(user_id, movie_id)
                predicted_rating = pred_result['predicted_rating']
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
                
            except Exception:
                continue
        
        if len(predictions) > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  Samples: {len(predictions)}")
        else:
            print("  ❌ No successful predictions")
        
        # 3. Recommendation Quality Sample
        print(f"\n🎯 RECOMMENDATION QUALITY SAMPLE")
        print("-" * 30)
        
        sample_user = test_users[0]
        
        try:
            # Get user's actual ratings
            user_ratings = engine.ratings_df[engine.ratings_df['userId'] == sample_user]
            high_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()
            
            print(f"  User {sample_user} has rated {len(high_rated_movies)} movies 4+ stars")
            
            # Get recommendations
            recommendations = engine.get_hybrid_recommendations(sample_user, 10)
            
            if recommendations:
                rec_movie_ids = [rec['movieId'] for rec in recommendations]
                hits = len(set(rec_movie_ids) & set(high_rated_movies))
                precision = hits / len(rec_movie_ids)
                
                print(f"  Recommendations: {len(recommendations)}")
                print(f"  Hits (movies user liked): {hits}")
                print(f"  Precision@10: {precision:.3f}")
                
                print(f"\n  Sample recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    title = rec.get('title', f"Movie {rec.get('movieId')}")
                    rating = rec.get('predicted_rating', 'N/A')
                    print(f"    {i}. {title} ({rating:.2f}⭐)")
            else:
                print("  ❌ No recommendations generated")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        # 4. New User Performance
        print(f"\n🆕 NEW USER PERFORMANCE")
        print("-" * 30)
        
        new_user_methods = ['popularity', 'genre', 'popularity_genre']
        
        for method in new_user_methods:
            try:
                start = time.time()
                
                if method == 'popularity':
                    recs = engine.get_new_user_recommendations(
                        user_preferences=None,
                        preferred_genres=None,
                        n_recommendations=5,
                        method=method
                    )
                else:
                    recs = engine.get_new_user_recommendations(
                        user_preferences=None,
                        preferred_genres=['Action', 'Comedy'],
                        n_recommendations=5,
                        method=method
                    )
                
                end = time.time()
                
                if recs:
                    print(f"  {method:15}: {len(recs)} recs in {end-start:.3f}s")
                else:
                    print(f"  {method:15}: No recommendations")
                    
            except Exception as e:
                print(f"  {method:15}: Error - {e}")
        
        # 5. Coverage Analysis
        print(f"\n🌐 COVERAGE ANALYSIS")
        print("-" * 30)
        
        try:
            # Test coverage with sample users
            all_recommended_items = set()
            
            for user_id in test_users[:3]:  # Use fewer users for quick test
                try:
                    recs = engine.get_hybrid_recommendations(user_id, 10)
                    if recs:
                        rec_items = [rec['movieId'] for rec in recs]
                        all_recommended_items.update(rec_items)
                except Exception:
                    continue
            
            total_movies = len(engine.movies_df) if engine.movies_df is not None else 27278
            coverage = len(all_recommended_items) / total_movies
            
            print(f"  Total movies in catalog: {total_movies:,}")
            print(f"  Unique movies recommended: {len(all_recommended_items)}")
            print(f"  Catalog coverage: {coverage:.4f} ({coverage*100:.2f}%)")
            
        except Exception as e:
            print(f"  ❌ Coverage analysis failed: {e}")
        
        print(f"\n✅ Quick evaluation completed!")
        print(f"\n💡 Tips:")
        print(f"   - Lower RMSE/MAE = better rating prediction")
        print(f"   - Higher Precision@10 = better recommendation quality")
        print(f"   - Higher coverage = more diverse recommendations")
        print(f"   - For detailed analysis, run: python evaluate_system.py")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_evaluation()
