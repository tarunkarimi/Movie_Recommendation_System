# Movie Recommendation System - Setup Guide 🎬

## 🚀 Quick Start

### 1. Install Dependencies

**Option A: Direct Installation (Recommended)**
```bash
# Install core build tools first
pip install --upgrade setuptools wheel pip

# Install packages individually for better compatibility
pip install numpy pandas
pip install scikit-learn scipy
pip install xgboost lightgbm
pip install matplotlib seaborn plotly
pip install jupyter ipykernel tqdm joblib
```

**Option B: From Requirements File**
```bash
pip install -r requirements.txt
```

**If you encounter build errors:**
1. Make sure you have Microsoft C++ Build Tools installed
2. Try installing packages individually as shown in Option A
3. Use `pip install --no-cache-dir package_name` for problematic packages

### 2. Download MovieLens Dataset
1. Visit: https://grouplens.org/datasets/movielens/20m/
2. Download `ml-20m.zip`
3. Extract `ratings.csv` and `movies.csv` to `data/raw/`

### 3. Run the System
```bash
# Quick demo (works without data)
python demo.py

# Web Interface - Interactive Dashboard
python launch_app.py
# This will open http://localhost:8501 in your browser

# Or launch Streamlit directly
streamlit run streamlit_app.py

# Full pipeline
python main.py --action demo
```

## 🌐 Web Interface (Streamlit App)

The system includes a comprehensive web dashboard built with Streamlit:

### Features:
- **📊 System Overview**: Real-time metrics and data visualization
- **🎯 Interactive Recommendations**: Generate personalized recommendations
- **👤 User Profile Analysis**: Detailed user behavior insights  
- **📈 Analytics Dashboard**: Advanced data exploration and insights
- **🎨 Modern UI**: Netflix-inspired responsive design

### Launch Options:

**Option 1: Easy Launcher (Recommended)**
```bash
python launch_app.py
```

**Option 2: Direct Streamlit Command**
```bash
streamlit run streamlit_app.py
```

**Option 3: Custom Configuration**
```bash
streamlit run streamlit_app.py --server.port 8502 --server.headless false
```

### Accessing the App:
- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501 (for sharing on local network)
- **Automatic Browser**: The launcher will open your default browser automatically

### Web Interface Navigation:

1. **🏠 Overview Tab**:
   - System statistics and metrics
   - Rating distribution charts
   - Genre popularity analysis
   - Data quality indicators

2. **🎯 Recommendations Tab**:
   - User selection dropdown
   - Algorithm choice (Hybrid/Similarity/ML)
   - Number of recommendations slider
   - Real-time recommendation generation
   - Detailed recommendation explanations

3. **👤 User Profile Tab**:
   - Individual user analysis
   - Rating history and patterns
   - User behavior insights
   - Preference visualization

4. **📈 Analytics Tab**:
   - Rating trends over time
   - Movie popularity analysis
   - Genre performance metrics
   - User behavior patterns

### Troubleshooting Web Interface:

1. **Port Already in Use**:
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

2. **Browser Doesn't Open**:
   - Manually navigate to http://localhost:8501
   - Check firewall settings
   - Try different browser

3. **Slow Loading**:
   - Large datasets may take time to load
   - Consider using sample data for faster performance
   - Check system memory usage

## 📊 Complete Pipeline

### Step 1: Data Preprocessing
```bash
python main.py --action preprocess
```
This will:
- Load and validate the MovieLens dataset
- Clean and filter the data
- Create user-item matrices
- Generate data statistics

### Step 2: Train Models
```bash
python main.py --action train
```
This will:
- Create engineered features
- Train multiple ML models (Linear Regression, Random Forest, XGBoost, etc.)
- Evaluate model performance
- Save the best models

### Step 3: Generate Recommendations
```bash
# Get recommendations for a specific user
python main.py --action recommend --user-id 123

# Use specific method
python main.py --action recommend --user-id 123 --method ml

# Get more recommendations
python main.py --action recommend --user-id 123 --n-recs 20
```

## 📁 Project Structure

```
Movie_rec/
├── data/
│   ├── raw/                    # Original MovieLens data
│   │   ├── ratings.csv        # User ratings (download required)
│   │   └── movies.csv         # Movie metadata (download required)
│   └── processed/             # Preprocessed data
│       ├── filtered_ratings.csv
│       ├── ml_features.csv
│       └── data_statistics.csv
├── src/                       # Source code modules
│   ├── data_preprocessing.py  # Data loading and cleaning
│   ├── similarity_engine.py   # Similarity calculations
│   ├── collaborative_filtering.py  # CF algorithms
│   ├── feature_engineering.py # Feature creation
│   ├── ml_models.py          # ML model training
│   └── recommendation_engine.py  # Main engine
├── models/                    # Trained models
│   ├── xgboost_model.pkl
│   ├── random_forest_model.pkl
│   └── feature_scaler.pkl
├── notebooks/                 # Jupyter analysis
│   └── movie_recommendation_analysis.ipynb
├── main.py                   # Main application
├── demo.py                   # Quick demo script
└── requirements.txt          # Dependencies
```

## 🔧 API Usage

### Using the Recommendation Engine
```python
from src.recommendation_engine import MovieRecommendationEngine

# Initialize
engine = MovieRecommendationEngine()
engine.load_data()

# Get recommendations
recommendations = engine.get_hybrid_recommendations(user_id=123, n_recommendations=10)

# Predict rating
prediction = engine.predict_user_rating(user_id=123, movie_id=1)
print(f"Predicted rating: {prediction['predicted_rating']:.2f}")
```

### Using Individual Components
```python
# Data preprocessing
from src.data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
ratings_df, movies_df = preprocessor.load_data()

# Feature engineering
from src.feature_engineering import FeatureEngineer
feature_engineer = FeatureEngineer(ratings_df)
features_df = feature_engineer.create_all_features(movies_df)

# ML training
from src.ml_models import MLModelTrainer
trainer = MLModelTrainer(features_df)
trainer.train_all_models()
```

## 📊 Expected Performance

### Model Performance (with full dataset)
- **XGBoost**: RMSE ≈ 0.83, MAPE ≈ 21.8%
- **Random Forest**: RMSE ≈ 0.85, MAPE ≈ 22.5%
- **Linear Regression**: RMSE ≈ 0.90, MAPE ≈ 25.0%

### System Capabilities
- ✅ Handles 20M+ ratings
- ✅ Supports 138K+ users
- ✅ Covers 27K+ movies
- ✅ Real-time recommendations
- ✅ Cold-start user handling
- ✅ Hybrid approach combining multiple methods

## 🎯 Recommendation Methods

### 1. Collaborative Filtering
- **Item-based**: Finds similar movies based on user rating patterns
- **User-based**: Finds similar users with comparable tastes

### 2. Machine Learning
- **Features**: 25+ engineered features including user bias, movie popularity, etc.
- **Models**: XGBoost, Random Forest, Linear models
- **Accuracy**: RMSE < 0.85 target

### 3. Hybrid System
- **Combination**: Weighted combination of CF and ML approaches
- **Flexibility**: Adjustable weights for different scenarios
- **Robustness**: Fallback mechanisms for edge cases

## 🧪 Testing and Evaluation

### Metrics Used
- **RMSE** (Root Mean Squared Error): Lower is better
- **MAE** (Mean Absolute Error): Lower is better  
- **MAPE** (Mean Absolute Percentage Error): Lower is better

### Cross-Validation
```python
# 5-fold cross-validation
trainer.cross_validate_model('XGBoost', cv_folds=5)
```

### A/B Testing Framework
```python
# Compare different recommendation methods
methods = ['similarity', 'ml', 'hybrid']
for method in methods:
    recommendations = engine.recommend_movies(user_id, method=method)
    # Evaluate recommendation quality
```

## 🔮 Future Enhancements

### Deep Learning Integration
- Neural Collaborative Filtering (NCF)
- Wide & Deep Learning
- Autoencoders for dimensionality reduction

### Advanced Features
- Real-time learning
- Context-aware recommendations
- Multi-criteria rating prediction
- Temporal dynamics modeling

### Scalability Improvements
- Distributed computing with Spark
- Online learning algorithms
- Caching and optimization
- API service deployment

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Not Found**
   ```
   Download MovieLens 20M dataset to data/raw/
   ```

3. **Memory Issues**
   ```python
   # Reduce dataset size in data_preprocessing.py
   MIN_USER_RATINGS = 50  # Increase filter thresholds
   MIN_MOVIE_RATINGS = 20
   ```

4. **Model Training Fails**
   ```bash
   # Install XGBoost
   pip install xgboost
   
   # Or use basic models only
   python main.py --action train --models-only basic
   ```

### Performance Tuning

1. **Speed up similarity calculations**
   ```python
   # Use sampling in similarity_engine.py
   user_similarity_df = calculate_user_similarity(
       user_item_matrix, sample_size=1000
   )
   ```

2. **Optimize feature engineering**
   ```python
   # Reduce feature complexity
   features_df = feature_engineer.create_basic_features()  # Skip advanced features
   ```

## 📚 References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Collaborative Filtering Techniques](https://en.wikipedia.org/wiki/Collaborative_filtering)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-4899-7637-6)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
