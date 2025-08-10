"""
Comprehensive Evaluation Script for Movie Recommendation System

This script provides various evaluation metrics for testing the performance
of different recommendation approaches including:
- Rating Prediction Accuracy (RMSE, MAE, MAPE)
- Recommendation Quality (Precision, Recall, F1, NDCG)
- Coverage and Diversity Metrics
- New User Performance
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

class RecommendationEvaluator:
    """Comprehensive evaluation class for recommendation systems."""
    
    def __init__(self, engine=None):
        """Initialize evaluator with recommendation engine."""
        self.engine = engine
        self.test_users = None
        self.test_data = None
        
    def split_data_for_evaluation(self, ratings_df: pd.DataFrame, test_ratio: float = 0.2):
        """Split ratings data for evaluation."""
        print(f"📊 Splitting data for evaluation ({test_ratio:.0%} for testing)...")
        
        # Get users with sufficient ratings for testing
        user_counts = ratings_df['userId'].value_counts()
        min_ratings_threshold = max(5, 10)  # Lower threshold for smaller datasets
        users_with_enough_ratings = user_counts[user_counts >= min_ratings_threshold].index
        
        print(f"Found {len(users_with_enough_ratings)} users with {min_ratings_threshold}+ ratings")
        
        if len(users_with_enough_ratings) == 0:
            # If still no users, lower the threshold further
            min_ratings_threshold = 3
            users_with_enough_ratings = user_counts[user_counts >= min_ratings_threshold].index
            print(f"Lowered threshold: Found {len(users_with_enough_ratings)} users with {min_ratings_threshold}+ ratings")
        
        if len(users_with_enough_ratings) == 0:
            raise ValueError("No users found with sufficient ratings for evaluation")
        
        # Sample test users
        n_test_users = min(500, len(users_with_enough_ratings) // 2, len(users_with_enough_ratings))
        if n_test_users == 0:
            n_test_users = min(len(users_with_enough_ratings), 100)
            
        self.test_users = np.random.choice(users_with_enough_ratings, n_test_users, replace=False)
        
        # For each test user, split their ratings
        train_data = []
        test_data = []
        
        for user_id in self.test_users:
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            # Skip users with too few ratings
            if len(user_ratings) < min_ratings_threshold:
                continue
            
            # Split user's ratings
            if len(user_ratings) >= 4:  # Need at least 4 ratings to split properly
                train_user, test_user = train_test_split(
                    user_ratings, test_size=test_ratio, random_state=42
                )
                
                train_data.append(train_user)
                test_data.append(test_user)
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise ValueError("No valid train/test splits created")
        
        self.train_data = pd.concat(train_data, ignore_index=True)
        self.test_data = pd.concat(test_data, ignore_index=True)
        
        print(f"✅ Train set: {len(self.train_data):,} ratings")
        print(f"✅ Test set: {len(self.test_data):,} ratings")
        
        return self.train_data, self.test_data
    
    def evaluate_rating_prediction(self, method: str = "hybrid", n_predictions: int = 1000) -> Dict[str, float]:
        """Evaluate rating prediction accuracy."""
        print(f"\n📈 Evaluating Rating Prediction Accuracy ({method})...")
        
        if self.test_data is None:
            raise ValueError("Must call split_data_for_evaluation first")
        
        # Sample test cases
        test_sample = self.test_data.sample(min(n_predictions, len(self.test_data)), random_state=42)
        
        predictions = []
        actuals = []
        errors = 0
        
        start_time = time.time()
        
        for _, row in test_sample.iterrows():
            try:
                user_id = row['userId']
                movie_id = row['movieId']
                actual_rating = row['rating']
                
                if method == "hybrid":
                    recs = self.engine.get_hybrid_recommendations(user_id, 1)
                elif method == "similarity":
                    recs = self.engine.recommend_movies_similarity(user_id, 'item_based', 1)
                elif method == "ml":
                    recs = self.engine.recommend_movies_ml(user_id, 'XGBoost', 1)
                else:
                    # Use direct prediction if available
                    pred_result = self.engine.predict_user_rating(user_id, movie_id)
                    predicted_rating = pred_result['predicted_rating']
                    
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
                    continue
                
                # For recommendation methods, estimate rating
                if recs and len(recs) > 0:
                    predicted_rating = recs[0].get('predicted_rating', self.engine.global_mean)
                else:
                    predicted_rating = self.engine.global_mean
                
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
                
            except Exception as e:
                errors += 1
                continue
        
        eval_time = time.time() - start_time
        
        if len(predictions) == 0:
            return {"error": "No successful predictions"}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Additional metrics
        correlation = np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else 0
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Correlation': correlation,
            'Predictions': len(predictions),
            'Errors': errors,
            'Time': eval_time
        }
        
        return metrics
    
    def evaluate_recommendation_quality(self, method: str = "hybrid", k: int = 10) -> Dict[str, float]:
        """Evaluate recommendation quality using top-K metrics."""
        print(f"\n🎯 Evaluating Recommendation Quality ({method}, top-{k})...")
        
        if self.test_data is None:
            raise ValueError("Must call split_data_for_evaluation first")
        
        precisions = []
        recalls = []
        ndcgs = []
        hit_rates = []
        
        # Sample test users
        test_users_sample = np.random.choice(self.test_users, min(100, len(self.test_users)), replace=False)
        
        start_time = time.time()
        
        for user_id in test_users_sample:
            try:
                # Get user's test ratings (relevant items)
                user_test_ratings = self.test_data[self.test_data['userId'] == user_id]
                relevant_items = set(user_test_ratings[user_test_ratings['rating'] >= 4.0]['movieId'])
                
                if len(relevant_items) == 0:
                    continue
                
                # Get recommendations
                if method == "hybrid":
                    recs = self.engine.get_hybrid_recommendations(user_id, k)
                elif method == "similarity":
                    recs = self.engine.recommend_movies_similarity(user_id, 'item_based', k)
                elif method == "ml":
                    recs = self.engine.recommend_movies_ml(user_id, 'XGBoost', k)
                else:
                    continue
                
                if not recs:
                    continue
                
                recommended_items = [rec['movieId'] for rec in recs]
                
                # Calculate metrics
                hits = len(set(recommended_items) & relevant_items)
                
                # Precision@K
                precision = hits / k if k > 0 else 0
                precisions.append(precision)
                
                # Recall@K
                recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0
                recalls.append(recall)
                
                # Hit Rate@K
                hit_rate = 1 if hits > 0 else 0
                hit_rates.append(hit_rate)
                
                # NDCG@K (simplified version)
                dcg = 0
                idcg = sum([1/np.log2(i+2) for i in range(min(len(relevant_items), k))])
                
                for i, item_id in enumerate(recommended_items):
                    if item_id in relevant_items:
                        dcg += 1 / np.log2(i + 2)
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
                
            except Exception as e:
                continue
        
        eval_time = time.time() - start_time
        
        metrics = {
            f'Precision@{k}': np.mean(precisions) if precisions else 0,
            f'Recall@{k}': np.mean(recalls) if recalls else 0,
            f'F1@{k}': 2 * np.mean(precisions) * np.mean(recalls) / (np.mean(precisions) + np.mean(recalls)) if (precisions and recalls and (np.mean(precisions) + np.mean(recalls)) > 0) else 0,
            f'HitRate@{k}': np.mean(hit_rates) if hit_rates else 0,
            f'NDCG@{k}': np.mean(ndcgs) if ndcgs else 0,
            'Users_Evaluated': len(precisions),
            'Time': eval_time
        }
        
        return metrics
    
    def evaluate_coverage_and_diversity(self, method: str = "hybrid", n_users: int = 100, k: int = 10) -> Dict[str, float]:
        """Evaluate catalog coverage and recommendation diversity."""
        print(f"\n🌐 Evaluating Coverage and Diversity ({method})...")
        
        all_recommended_items = set()
        user_recommendations = []
        
        # Sample users for diversity evaluation
        sample_users = np.random.choice(self.test_users, min(n_users, len(self.test_users)), replace=False)
        
        start_time = time.time()
        
        for user_id in sample_users:
            try:
                if method == "hybrid":
                    recs = self.engine.get_hybrid_recommendations(user_id, k)
                elif method == "similarity":
                    recs = self.engine.recommend_movies_similarity(user_id, 'item_based', k)
                elif method == "ml":
                    recs = self.engine.recommend_movies_ml(user_id, 'XGBoost', k)
                else:
                    continue
                
                if recs:
                    rec_items = [rec['movieId'] for rec in recs]
                    user_recommendations.append(rec_items)
                    all_recommended_items.update(rec_items)
                    
            except Exception:
                continue
        
        eval_time = time.time() - start_time
        
        # Calculate metrics
        total_items = len(self.engine.movies_df) if self.engine.movies_df is not None else 27278
        catalog_coverage = len(all_recommended_items) / total_items
        
        # Calculate intra-list diversity (average pairwise diversity within recommendations)
        intra_diversities = []
        for rec_list in user_recommendations:
            if len(rec_list) > 1:
                diversity = len(set(rec_list)) / len(rec_list)  # Simplified diversity
                intra_diversities.append(diversity)
        
        # Calculate personalization (how different are recommendations between users)
        if len(user_recommendations) > 1:
            all_pairs_intersection = []
            for i in range(len(user_recommendations)):
                for j in range(i+1, len(user_recommendations)):
                    intersection = len(set(user_recommendations[i]) & set(user_recommendations[j]))
                    union = len(set(user_recommendations[i]) | set(user_recommendations[j]))
                    jaccard = intersection / union if union > 0 else 0
                    all_pairs_intersection.append(jaccard)
            
            personalization = 1 - np.mean(all_pairs_intersection)  # Lower similarity = higher personalization
        else:
            personalization = 0
        
        metrics = {
            'Catalog_Coverage': catalog_coverage,
            'Items_Recommended': len(all_recommended_items),
            'Intra_List_Diversity': np.mean(intra_diversities) if intra_diversities else 0,
            'Personalization': personalization,
            'Users_Analyzed': len(user_recommendations),
            'Time': eval_time
        }
        
        return metrics
    
    def evaluate_new_user_performance(self, n_tests: int = 50) -> Dict[str, Any]:
        """Evaluate performance for new users (cold start)."""
        print(f"\n🆕 Evaluating New User Performance...")
        
        methods = ['popularity', 'genre', 'popularity_genre']
        genre_combinations = [
            ['Action', 'Adventure'], 
            ['Comedy', 'Romance'], 
            ['Drama', 'Thriller'],
            ['Sci-Fi', 'Fantasy'],
            ['Horror', 'Mystery']
        ]
        
        results = {}
        
        for method in methods:
            print(f"  Testing method: {method}")
            response_times = []
            recommendation_counts = []
            
            start_time = time.time()
            
            for i in range(n_tests):
                try:
                    if method == 'popularity':
                        recs = self.engine.get_new_user_recommendations(
                            user_preferences=None,
                            preferred_genres=None,
                            n_recommendations=10,
                            method=method
                        )
                    else:
                        genres = genre_combinations[i % len(genre_combinations)]
                        recs = self.engine.get_new_user_recommendations(
                            user_preferences=None,
                            preferred_genres=genres,
                            n_recommendations=10,
                            method=method
                        )
                    
                    recommendation_counts.append(len(recs) if recs else 0)
                    
                except Exception:
                    recommendation_counts.append(0)
            
            total_time = time.time() - start_time
            avg_response_time = total_time / n_tests
            
            results[method] = {
                'Success_Rate': sum(1 for count in recommendation_counts if count > 0) / n_tests,
                'Avg_Recommendations': np.mean(recommendation_counts),
                'Avg_Response_Time': avg_response_time,
                'Total_Tests': n_tests
            }
        
        return results

def run_comprehensive_evaluation():
    """Run a comprehensive evaluation of the recommendation system."""
    print("🎬 Movie Recommendation System - Comprehensive Evaluation")
    print("=" * 60)
    
    try:
        # Load recommendation engine
        from recommendation_engine import MovieRecommendationEngine
        
        print("🔧 Loading recommendation engine...")
        engine = MovieRecommendationEngine()
        engine.load_data()
        print("✅ Engine loaded successfully!")
        
        # Initialize evaluator
        evaluator = RecommendationEvaluator(engine)
        
        # Split data for evaluation
        evaluator.split_data_for_evaluation(engine.ratings_df, test_ratio=0.2)
        
        # 1. Rating Prediction Evaluation
        print(f"\n{'='*60}")
        print("📈 RATING PREDICTION ACCURACY")
        print(f"{'='*60}")
        
        methods = ["hybrid", "similarity", "ml"]
        rating_results = {}
        
        for method in methods:
            print(f"\n🔍 Testing {method.upper()} method...")
            metrics = evaluator.evaluate_rating_prediction(method, n_predictions=500)
            rating_results[method] = metrics
            
            if 'error' not in metrics:
                print(f"  RMSE: {metrics['RMSE']:.4f}")
                print(f"  MAE:  {metrics['MAE']:.4f}")
                print(f"  MAPE: {metrics['MAPE']:.2f}%")
                print(f"  Correlation: {metrics['Correlation']:.4f}")
                print(f"  Time: {metrics['Time']:.2f}s")
            else:
                print(f"  ❌ {metrics['error']}")
        
        # 2. Recommendation Quality Evaluation
        print(f"\n{'='*60}")
        print("🎯 RECOMMENDATION QUALITY")
        print(f"{'='*60}")
        
        quality_results = {}
        
        for method in methods:
            print(f"\n🔍 Testing {method.upper()} method...")
            metrics = evaluator.evaluate_recommendation_quality(method, k=10)
            quality_results[method] = metrics
            
            print(f"  Precision@10: {metrics['Precision@10']:.4f}")
            print(f"  Recall@10:    {metrics['Recall@10']:.4f}")
            print(f"  F1@10:        {metrics['F1@10']:.4f}")
            print(f"  HitRate@10:   {metrics['HitRate@10']:.4f}")
            print(f"  NDCG@10:      {metrics['NDCG@10']:.4f}")
            print(f"  Users:        {metrics['Users_Evaluated']}")
        
        # 3. Coverage and Diversity Evaluation
        print(f"\n{'='*60}")
        print("🌐 COVERAGE AND DIVERSITY")
        print(f"{'='*60}")
        
        diversity_results = {}
        
        for method in methods:
            print(f"\n🔍 Testing {method.upper()} method...")
            metrics = evaluator.evaluate_coverage_and_diversity(method, n_users=50, k=10)
            diversity_results[method] = metrics
            
            print(f"  Catalog Coverage:     {metrics['Catalog_Coverage']:.4f}")
            print(f"  Items Recommended:    {metrics['Items_Recommended']}")
            print(f"  Intra-List Diversity: {metrics['Intra_List_Diversity']:.4f}")
            print(f"  Personalization:      {metrics['Personalization']:.4f}")
            print(f"  Users Analyzed:       {metrics['Users_Analyzed']}")
        
        # 4. New User Performance
        print(f"\n{'='*60}")
        print("🆕 NEW USER PERFORMANCE")
        print(f"{'='*60}")
        
        new_user_results = evaluator.evaluate_new_user_performance(n_tests=30)
        
        for method, metrics in new_user_results.items():
            print(f"\n🔍 {method.upper()} method:")
            print(f"  Success Rate:      {metrics['Success_Rate']:.2%}")
            print(f"  Avg Recommendations: {metrics['Avg_Recommendations']:.1f}")
            print(f"  Avg Response Time:   {metrics['Avg_Response_Time']:.3f}s")
        
        # 5. Summary Report
        print(f"\n{'='*60}")
        print("📊 EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print("\n🥇 Best Performing Methods:")
        
        # Best RMSE
        best_rmse = min(rating_results.items(), key=lambda x: x[1].get('RMSE', float('inf')))
        print(f"  Rating Prediction (RMSE): {best_rmse[0].upper()} ({best_rmse[1].get('RMSE', 'N/A'):.4f})")
        
        # Best Precision
        best_precision = max(quality_results.items(), key=lambda x: x[1].get('Precision@10', 0))
        print(f"  Recommendation Quality:   {best_precision[0].upper()} (Precision@10: {best_precision[1].get('Precision@10', 'N/A'):.4f})")
        
        # Best Coverage
        best_coverage = max(diversity_results.items(), key=lambda x: x[1].get('Catalog_Coverage', 0))
        print(f"  Coverage:                 {best_coverage[0].upper()} ({best_coverage[1].get('Catalog_Coverage', 'N/A'):.4f})")
        
        # Best New User Success
        best_new_user = max(new_user_results.items(), key=lambda x: x[1].get('Success_Rate', 0))
        print(f"  New User Performance:     {best_new_user[0].upper()} ({best_new_user[1].get('Success_Rate', 0):.2%} success)")
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📝 Consider the trade-offs between accuracy, coverage, and diversity when choosing methods.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_evaluation()
