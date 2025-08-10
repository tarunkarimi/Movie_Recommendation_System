# Movie Recommendation System - Evaluation Guide

This guide explains how to evaluate the performance of your movie recommendation system using various metrics and approaches.

## 📊 Evaluation Scripts

### 1. Quick Evaluation (`quick_eval.py`)
**Purpose**: Fast overview of system performance  
**Runtime**: ~2-3 minutes  
**Usage**:
```bash
python quick_eval.py
# OR
python demo.py --evaluate
```

**Metrics Covered**:
- ⏱️ **Response Time**: Average time to generate recommendations
- 📈 **Rating Prediction**: RMSE, MAE, MAPE for rating accuracy  
- 🎯 **Quality Sample**: Precision@10 for a sample user
- 🆕 **New User Performance**: Success rates for cold-start scenarios
- 🌐 **Coverage**: Catalog coverage analysis

### 2. Comprehensive Evaluation (`evaluate_system.py`)
**Purpose**: Detailed analysis with statistical significance  
**Runtime**: ~15-30 minutes  
**Usage**:
```bash
python evaluate_system.py
```

**Metrics Covered**:
- 📈 **Rating Prediction Accuracy**: RMSE, MAE, MAPE, Correlation
- 🎯 **Recommendation Quality**: Precision@K, Recall@K, F1@K, HitRate@K, NDCG@K
- 🌐 **Coverage & Diversity**: Catalog coverage, Intra-list diversity, Personalization
- 🆕 **New User Performance**: Success rates, response times across methods

## 📋 Evaluation Metrics Explained

### Rating Prediction Metrics
- **RMSE (Root Mean Square Error)**: Lower is better (0-5 scale)
  - Excellent: < 0.8
  - Good: 0.8-1.0  
  - Acceptable: 1.0-1.2
  - Poor: > 1.2

- **MAE (Mean Absolute Error)**: Lower is better (0-5 scale)
  - Excellent: < 0.6
  - Good: 0.6-0.8
  - Acceptable: 0.8-1.0
  - Poor: > 1.0

- **MAPE (Mean Absolute Percentage Error)**: Lower is better
  - Excellent: < 20%
  - Good: 20-30%
  - Acceptable: 30-50%
  - Poor: > 50%

### Recommendation Quality Metrics
- **Precision@K**: Fraction of recommended items that are relevant
  - Formula: (Relevant items in top-K) / K
  - Higher is better (0-1 scale)

- **Recall@K**: Fraction of relevant items that are recommended
  - Formula: (Relevant items in top-K) / (Total relevant items)
  - Higher is better (0-1 scale)

- **F1@K**: Harmonic mean of Precision and Recall
  - Formula: 2 × (Precision × Recall) / (Precision + Recall)
  - Higher is better (0-1 scale)

- **HitRate@K**: Fraction of users with at least one relevant item in top-K
  - Higher is better (0-1 scale)

- **NDCG@K (Normalized Discounted Cumulative Gain)**: Ranking quality metric
  - Considers both relevance and position
  - Higher is better (0-1 scale)

### Coverage & Diversity Metrics
- **Catalog Coverage**: Fraction of items ever recommended
  - Formula: (Unique recommended items) / (Total items)
  - Higher indicates better exploration of catalog

- **Intra-list Diversity**: Diversity within individual recommendation lists
  - Higher indicates more varied recommendations per user

- **Personalization**: How different recommendations are between users
  - Higher indicates better personalization

## 🎯 Benchmark Results

### Current System Performance (MovieLens 20M)
```
📈 RATING PREDICTION
  RMSE: ~1.12 (Acceptable - fallback to popular movies)
  MAE:  ~0.89 (Good)
  MAPE: ~48%  (Acceptable)

🎯 RECOMMENDATION QUALITY  
  Precision@10: ~0.30 (Good for cold users)
  Coverage: ~0.04% (Low - using popular movies fallback)

⏱️ RESPONSE TIMES
  Similarity: ~0.8s (Good)
  ML: ~0.8s (Good)  
  Hybrid: ~2.4s (Acceptable)
  New User: ~0.9-3.0s (Good to Acceptable)
```

## 🔧 Performance Optimization Tips

### 1. Improve Rating Prediction
- **Train ML models**: Run `python main.py --action train` to train XGBoost/RF models
- **Build similarity matrices**: Enable collaborative filtering for better similarity-based recommendations
- **Feature engineering**: Add more user/item features

### 2. Improve Recommendation Quality
- **Hybrid approaches**: Combine multiple methods for better results
- **Parameter tuning**: Optimize similarity thresholds, model hyperparameters
- **Data filtering**: Remove unreliable ratings, inactive users

### 3. Improve Coverage & Diversity
- **Recommendation diversification**: Add diversity boosting to ranking
- **Exploration strategies**: Balance exploitation vs exploration
- **Long-tail promotion**: Boost less popular but high-quality items

### 4. Improve Response Time
- **Caching**: Cache similarity matrices, popular items, user profiles
- **Preprocessing**: Precompute recommendations for active users
- **Model optimization**: Use faster models for real-time scenarios

## 📝 Interpreting Results

### What Good Performance Looks Like
```
✅ EXCELLENT SYSTEM:
   RMSE: < 0.8, Precision@10: > 0.4, Coverage: > 5%, Response: < 1s

✅ GOOD SYSTEM:
   RMSE: 0.8-1.0, Precision@10: 0.3-0.4, Coverage: 1-5%, Response: 1-2s

⚠️ NEEDS IMPROVEMENT:
   RMSE: > 1.2, Precision@10: < 0.2, Coverage: < 1%, Response: > 5s
```

### Trade-offs to Consider
- **Accuracy vs Speed**: More complex models = better accuracy but slower response
- **Personalization vs Coverage**: Highly personalized = lower coverage of catalog
- **Popularity vs Discovery**: Popular items = higher precision but less discovery

## 🚀 Running Full Evaluation Pipeline

For complete system evaluation:

```bash
# 1. Ensure data is loaded
python main.py --action preprocess

# 2. Train models (optional but recommended)
python main.py --action train

# 3. Quick evaluation
python quick_eval.py

# 4. Comprehensive evaluation  
python evaluate_system.py

# 5. Compare with baseline
python demo.py --evaluate
```

## 📊 Custom Evaluation

To create custom evaluations, use the `RecommendationEvaluator` class:

```python
from evaluate_system import RecommendationEvaluator
from recommendation_engine import MovieRecommendationEngine

# Load engine
engine = MovieRecommendationEngine()
engine.load_data()

# Create evaluator
evaluator = RecommendationEvaluator(engine)

# Split data
evaluator.split_data_for_evaluation(engine.ratings_df)

# Run specific evaluations
rating_metrics = evaluator.evaluate_rating_prediction("hybrid")
quality_metrics = evaluator.evaluate_recommendation_quality("hybrid", k=10)
```

## 📈 Monitoring in Production

For production systems, monitor:
- **Response times** (SLA compliance)
- **Recommendation diversity** (avoid filter bubbles)  
- **User engagement** (click-through rates, conversion)
- **Coverage decay** (ensure new items get recommended)
- **A/B testing** (compare algorithm variations)

---

📚 **References**: 
- Ricci, F., et al. "Recommender Systems Handbook" (2015)
- Aggarwal, C. "Recommender Systems: The Textbook" (2016)
- MovieLens evaluation methodologies
