"""
Test script to demonstrate evaluation metrics functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

def test_evaluation_metrics():
    """Test all evaluation functionality."""
    print("🧪 Testing Evaluation Metrics")
    print("=" * 40)
    
    try:
        from recommendation_engine import MovieRecommendationEngine
        from evaluate_system import RecommendationEvaluator
        import time
        
        # Load engine
        print("🔧 Loading recommendation engine...")
        engine = MovieRecommendationEngine()
        engine.load_data()
        print("✅ Engine loaded!")
        
        # Create evaluator
        evaluator = RecommendationEvaluator(engine)
        
        # Test 1: Data splitting
        print("\n📊 Testing data splitting...")
        sample_data = engine.ratings_df.sample(10000, random_state=42)
        train_data, test_data = evaluator.split_data_for_evaluation(sample_data, test_ratio=0.2)
        print(f"✅ Split successful: {len(train_data)} train, {len(test_data)} test")
        
        # Test 2: Rating prediction evaluation
        print("\n📈 Testing rating prediction evaluation...")
        start_time = time.time()
        rating_metrics = evaluator.evaluate_rating_prediction('hybrid', n_predictions=10)
        eval_time = time.time() - start_time
        
        print("✅ Rating prediction metrics:")
        for metric, value in rating_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        print(f"   Evaluation time: {eval_time:.2f}s")
        
        # Test 3: New user evaluation
        print("\n🆕 Testing new user evaluation...")
        new_user_metrics = evaluator.evaluate_new_user_performance(n_tests=5)
        
        print("✅ New user performance:")
        for method, metrics in new_user_metrics.items():
            success_rate = metrics['Success_Rate']
            avg_recs = metrics['Avg_Recommendations']
            print(f"   {method}: {success_rate:.2%} success, {avg_recs:.1f} avg recs")
        
        # Test 4: Basic recommendation quality
        print("\n🎯 Testing recommendation quality...")
        quality_metrics = evaluator.evaluate_recommendation_quality('hybrid', k=5)
        
        print("✅ Recommendation quality:")
        for metric, value in quality_metrics.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.4f}")
            else:
                print(f"   {metric}: {value}")
        
        print("\n🎉 All evaluation tests passed!")
        print("\n💡 To run full evaluations:")
        print("   python quick_eval.py          # Fast overview")
        print("   python evaluate_system.py     # Comprehensive analysis")
        print("   python demo.py --evaluate     # Demo with evaluation")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation_metrics()
