"""
🎬 Movie Recommendation System - Streamlit Web App

A comprehensive web interface for the movie recommendation system
featuring collaborative filtering and machine learning approaches.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #E50914;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E50914;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommendation_engine():
    """Load the recommendation engine with caching."""
    try:
        from src.recommendation_engine import MovieRecommendationEngine
        engine = MovieRecommendationEngine()
        engine.load_data()
        return engine, True
    except Exception as e:
        st.error(f"Error loading recommendation engine: {str(e)}")
        return None, False

@st.cache_resource
def load_sample_data():
    """Load or create sample data for demonstration."""
    try:
        # Try to load real data first
        if os.path.exists("data/raw/ratings.csv"):
            ratings_df = pd.read_csv("data/raw/ratings.csv")
            movies_df = pd.read_csv("data/raw/movies.csv") if os.path.exists("data/raw/movies.csv") else None
            return ratings_df, movies_df, True
        else:
            # Create synthetic data for demo
            np.random.seed(42)
            n_users, n_movies, n_ratings = 1000, 500, 10000
            
            # Generate synthetic ratings
            user_ids = np.random.choice(range(1, n_users + 1), n_ratings)
            movie_ids = np.random.choice(range(1, n_movies + 1), n_ratings)
            ratings = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.1, 0.1, 0.2, 0.3, 0.3])
            timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
            
            ratings_df = pd.DataFrame({
                'userId': user_ids,
                'movieId': movie_ids,
                'rating': ratings,
                'timestamp': timestamps
            }).drop_duplicates(subset=['userId', 'movieId'])
            
            # Generate synthetic movies
            genres_list = ['Action', 'Adventure', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
            movies_data = []
            
            for movie_id in range(1, n_movies + 1):
                n_genres = np.random.randint(1, 4)
                movie_genres = np.random.choice(genres_list, n_genres, replace=False)
                movies_data.append({
                    'movieId': movie_id,
                    'title': f'Movie_{movie_id} ({np.random.randint(1990, 2024)})',
                    'genres': '|'.join(movie_genres)
                })
            
            movies_df = pd.DataFrame(movies_data)
            return ratings_df, movies_df, False
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, False

def display_main_header():
    """Display the main application header."""
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-bottom: 2rem;">
            Production-ready recommendation system using collaborative filtering + machine learning
        </div>
        """, 
        unsafe_allow_html=True
    )

def display_sidebar_info(ratings_df, movies_df, is_real_data):
    """Display dataset information in the sidebar."""
    st.sidebar.markdown("## 📊 Dataset Information")
    
    if ratings_df is not None:
        st.sidebar.metric("👥 Total Users", f"{ratings_df['userId'].nunique():,}")
        st.sidebar.metric("🎬 Total Movies", f"{ratings_df['movieId'].nunique():,}")
        st.sidebar.metric("⭐ Total Ratings", f"{len(ratings_df):,}")
        st.sidebar.metric("📈 Avg Rating", f"{ratings_df['rating'].mean():.2f}")
        
        if is_real_data:
            st.sidebar.success("✅ Using MovieLens Dataset")
        else:
            st.sidebar.info("ℹ️ Using Synthetic Data")
            st.sidebar.caption("Download MovieLens data to use real dataset")

def display_system_overview(ratings_df, movies_df):
    """Display system overview and statistics."""
    st.header("📈 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Users",
            value=f"{ratings_df['userId'].nunique():,}",
            delta="Active users in system"
        )
    
    with col2:
        st.metric(
            label="Total Movies",
            value=f"{ratings_df['movieId'].nunique():,}",
            delta="Movies in catalog"
        )
    
    with col3:
        st.metric(
            label="Total Ratings",
            value=f"{len(ratings_df):,}",
            delta="User interactions"
        )
    
    with col4:
        st.metric(
            label="Avg Rating",
            value=f"{ratings_df['rating'].mean():.2f} ⭐",
            delta="Overall rating quality"
        )
    
    # Rating distribution chart
    st.subheader("📊 Rating Distribution")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index, 
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title="Distribution of Ratings",
            color=rating_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top genres if available
        if movies_df is not None and 'genres' in movies_df.columns:
            st.subheader("🎭 Top Genres")
            genre_counts = {}
            for genres_str in movies_df['genres'].dropna():
                if genres_str != '(no genres listed)':
                    for genre in genres_str.split('|'):
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:8]
            for genre, count in top_genres:
                st.metric(genre, count)

def display_recommendation_interface(engine, ratings_df, movies_df):
    """Display the main recommendation interface."""
    st.header("🎯 Get Movie Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Configuration")
        
        # User selection
        available_users = sorted(ratings_df['userId'].unique())
        selected_user = st.selectbox(
            "👤 Select User ID",
            available_users,
            index=min(50, len(available_users)-1) if len(available_users) > 50 else 0,
            help="Choose a user to get personalized recommendations"
        )
        
        # Number of recommendations
        n_recommendations = st.slider(
            "📊 Number of Recommendations",
            min_value=5,
            max_value=20,
            value=10,
            help="How many movie recommendations to generate"
        )
        
        # Recommendation method
        method = st.selectbox(
            "🧠 Recommendation Method",
            ["hybrid", "similarity", "ml"],
            index=0,
            help="Choose the recommendation algorithm"
        )
        
        # Method explanations
        method_explanations = {
            "hybrid": "🔗 Combines collaborative filtering + machine learning for optimal results",
            "similarity": "👥 Uses user-user and item-item collaborative filtering",
            "ml": "🤖 Uses machine learning models (XGBoost, Random Forest) for predictions"
        }
        st.info(method_explanations[method])
        
        # Generate recommendations button
        if st.button("🚀 Generate Recommendations", type="primary"):
            with st.spinner("🎬 Generating personalized recommendations..."):
                recommendations = get_recommendations(engine, selected_user, n_recommendations, method)
                st.session_state.recommendations = recommendations
                st.session_state.selected_user = selected_user
                st.session_state.method = method
    
    with col2:
        st.subheader("🎬 Recommended Movies")
        
        # Display recommendations if available
        if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
            display_recommendations(st.session_state.recommendations, movies_df)
        else:
            st.info("👆 Configure settings and click 'Generate Recommendations' to see personalized movie suggestions!")

@st.cache_data
def get_recommendations(_engine, user_id, n_recommendations, method):
    """Get recommendations from the engine with caching."""
    try:
        if _engine is None:
            # Fallback to synthetic recommendations
            return generate_synthetic_recommendations(user_id, n_recommendations)
        
        if method == "hybrid":
            return _engine.get_hybrid_recommendations(user_id, n_recommendations)
        elif method == "similarity":
            return _engine.recommend_movies_similarity(user_id, method='item_based', n_recommendations=n_recommendations)
        elif method == "ml":
            return _engine.recommend_movies_ml(user_id, model_name='XGBoost', n_recommendations=n_recommendations)
        else:
            return generate_synthetic_recommendations(user_id, n_recommendations)
            
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return generate_synthetic_recommendations(user_id, n_recommendations)

def generate_synthetic_recommendations(user_id, n_recommendations):
    """Generate synthetic recommendations for demo purposes."""
    np.random.seed(user_id)  # Consistent recommendations per user
    
    movie_titles = [
        "The Shawshank Redemption", "The Godfather", "The Dark Knight", "Pulp Fiction",
        "The Lord of the Rings", "Forrest Gump", "Inception", "The Matrix",
        "Goodfellas", "The Empire Strikes Back", "One Flew Over the Cuckoo's Nest",
        "Se7en", "The Silence of the Lambs", "Saving Private Ryan", "Terminator 2",
        "Schindler's List", "Casablanca", "The Green Mile", "Raiders of the Lost Ark",
        "Alien", "Blade Runner", "The Departed", "Heat", "Gladiator"
    ]
    
    recommendations = []
    selected_movies = np.random.choice(movie_titles, min(n_recommendations, len(movie_titles)), replace=False)
    
    for i, title in enumerate(selected_movies):
        recommendations.append({
            'movieId': 1000 + i,
            'title': title,
            'predicted_rating': np.random.uniform(3.5, 5.0),
            'confidence': np.random.uniform(0.7, 0.95),
            'reason': np.random.choice([
                "Similar users loved this movie",
                "Based on your rating history",
                "Popular in your favorite genre",
                "Highly rated by critics"
            ])
        })
    
    return recommendations

def display_recommendations(recommendations, movies_df):
    """Display the recommendation results."""
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3 = st.columns([0.5, 3, 1])
            
            with col1:
                st.markdown(f"**#{i}**")
            
            with col2:
                title = rec.get('title', f"Movie ID: {rec.get('movieId', 'Unknown')}")
                st.markdown(f"**{title}**")
                
                if 'reason' in rec:
                    st.caption(f"💡 {rec['reason']}")
            
            with col3:
                rating = rec.get('predicted_rating', 0)
                confidence = rec.get('confidence', 0)
                
                st.metric("Predicted Rating", f"{rating:.1f} ⭐")
                if confidence > 0:
                    st.caption(f"Confidence: {confidence:.0%}")
            
            st.divider()

def display_user_profile(ratings_df, movies_df, user_id):
    """Display user profile and rating history."""
    st.header(f"👤 User Profile: {user_id}")
    
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    
    if len(user_ratings) == 0:
        st.warning("No rating history found for this user.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Movies Rated", len(user_ratings))
        st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f} ⭐")
    
    with col2:
        st.metric("Highest Rating", f"{user_ratings['rating'].max():.0f} ⭐")
        st.metric("Lowest Rating", f"{user_ratings['rating'].min():.0f} ⭐")
    
    with col3:
        rating_std = user_ratings['rating'].std()
        st.metric("Rating Variance", f"{rating_std:.2f}")
        if rating_std < 0.8:
            st.caption("😊 Consistent rater")
        elif rating_std > 1.5:
            st.caption("🎭 Diverse taste")
        else:
            st.caption("⚖️ Balanced rater")
    
    # User's rating distribution
    st.subheader("📊 Your Rating Distribution")
    user_rating_counts = user_ratings['rating'].value_counts().sort_index()
    
    fig = px.bar(
        x=user_rating_counts.index,
        y=user_rating_counts.values,
        labels={'x': 'Rating Given', 'y': 'Number of Movies'},
        title=f"Rating Pattern for User {user_id}",
        color=user_rating_counts.values,
        color_continuous_scale='blues'
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent ratings
    st.subheader("🕐 Recent Ratings")
    if 'timestamp' in user_ratings.columns:
        recent_ratings = user_ratings.nlargest(10, 'timestamp')
    else:
        recent_ratings = user_ratings.tail(10)
    
    for _, rating in recent_ratings.iterrows():
        movie_title = "Unknown Movie"
        if movies_df is not None:
            movie_info = movies_df[movies_df['movieId'] == rating['movieId']]
            if not movie_info.empty:
                movie_title = movie_info.iloc[0]['title']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"🎬 {movie_title}")
        with col2:
            st.write(f"{'⭐' * int(rating['rating'])} ({rating['rating']})")

def display_analytics_dashboard(ratings_df, movies_df):
    """Display analytics and insights dashboard."""
    st.header("📈 Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["📊 Rating Analytics", "🎬 Movie Analytics", "👥 User Analytics"])
    
    with tab1:
        st.subheader("Rating Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution over time
            if 'timestamp' in ratings_df.columns:
                ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
                daily_ratings = ratings_df.groupby(ratings_df['date'].dt.date).size()
                
                fig = px.line(
                    x=daily_ratings.index,
                    y=daily_ratings.values,
                    title="Ratings Over Time",
                    labels={'x': 'Date', 'y': 'Number of Ratings'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average rating by movie count
            movie_stats = ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).round(2)
            movie_stats.columns = ['avg_rating', 'rating_count']
            
            fig = px.scatter(
                movie_stats,
                x='rating_count',
                y='avg_rating',
                title="Movie Rating vs Popularity",
                labels={'rating_count': 'Number of Ratings', 'avg_rating': 'Average Rating'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Movie Analytics")
        
        if movies_df is not None:
            # Genre analysis
            genre_ratings = {}
            for _, movie in movies_df.iterrows():
                if 'genres' in movie and pd.notna(movie['genres']) and movie['genres'] != '(no genres listed)':
                    movie_ratings = ratings_df[ratings_df['movieId'] == movie['movieId']]['rating']
                    if len(movie_ratings) > 0:
                        for genre in movie['genres'].split('|'):
                            if genre not in genre_ratings:
                                genre_ratings[genre] = []
                            genre_ratings[genre].extend(movie_ratings.tolist())
            
            if genre_ratings:
                genre_avg = {genre: np.mean(ratings) for genre, ratings in genre_ratings.items() if len(ratings) >= 10}
                
                if genre_avg:
                    fig = px.bar(
                        x=list(genre_avg.keys()),
                        y=list(genre_avg.values()),
                        title="Average Rating by Genre",
                        labels={'x': 'Genre', 'y': 'Average Rating'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("User Behavior Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User activity distribution
            user_activity = ratings_df.groupby('userId').size()
            
            fig = px.histogram(
                x=user_activity.values,
                nbins=30,
                title="User Activity Distribution",
                labels={'x': 'Number of Ratings per User', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # User rating behavior
            user_avg_ratings = ratings_df.groupby('userId')['rating'].mean()
            
            fig = px.histogram(
                x=user_avg_ratings.values,
                nbins=30,
                title="User Average Rating Distribution",
                labels={'x': 'Average Rating Given', 'y': 'Number of Users'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_new_user_interface(engine, ratings_df, movies_df):
    """Display interface for new users to get initial recommendations."""
    st.header("🆕 New User Recommendations")
    st.markdown("Welcome! Get personalized movie recommendations by sharing your preferences.")
    
    # Initialize session state for new user
    if 'new_user_ratings' not in st.session_state:
        st.session_state.new_user_ratings = {}
    if 'new_user_genres' not in st.session_state:
        st.session_state.new_user_genres = []
    
    tab1, tab2, tab3 = st.tabs(["🎬 Rate Movies", "🎭 Select Genres", "🎯 Get Recommendations"])
    
    with tab1:
        st.subheader("🎬 Rate Some Movies")
        st.markdown("Rate a few movies to help us understand your taste (optional but recommended)")
        
        # Get sample movies for rating
        if movies_df is not None and ratings_df is not None:
            # Get popular movies for new users to rate
            popular_movie_ids = get_popular_movies_for_rating(ratings_df, movies_df, 20)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Select movies you've seen and rate them:**")
                
                for movie_id in popular_movie_ids:
                    movie_info = movies_df[movies_df['movieId'] == movie_id]
                    if not movie_info.empty:
                        movie_title = movie_info.iloc[0]['title']
                        
                        # Create a unique key for each movie
                        rating_key = f"rating_{movie_id}"
                        
                        # Rating slider
                        rating = st.select_slider(
                            f"🎬 **{movie_title}**",
                            options=[0, 1, 2, 3, 4, 5],
                            value=0,
                            key=rating_key,
                            format_func=lambda x: "Not Seen" if x == 0 else f"{x} ⭐",
                            help="Rate this movie if you've seen it"
                        )
                        
                        # Update session state
                        if rating > 0:
                            st.session_state.new_user_ratings[movie_id] = rating
                        elif movie_id in st.session_state.new_user_ratings:
                            del st.session_state.new_user_ratings[movie_id]
            
            with col2:
                st.markdown("**Your Ratings:**")
                if st.session_state.new_user_ratings:
                    for movie_id, rating in st.session_state.new_user_ratings.items():
                        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
                        st.write(f"{'⭐' * rating} {movie_title}")
                    
                    avg_rating = np.mean(list(st.session_state.new_user_ratings.values()))
                    st.metric("Your Average Rating", f"{avg_rating:.1f} ⭐")
                else:
                    st.info("No ratings yet. Rate some movies above!")
        else:
            st.info("Movie data not available. You can still select genres in the next tab.")
    
    with tab2:
        st.subheader("🎭 Select Your Favorite Genres")
        st.markdown("Choose genres you enjoy to get better recommendations")
        
        # Available genres
        available_genres = [
            'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        # Genre selection with multiselect
        selected_genres = st.multiselect(
            "🎭 Select your favorite genres:",
            available_genres,
            default=st.session_state.new_user_genres,
            help="Choose 3-5 genres you enjoy most"
        )
        
        # Update session state immediately
        st.session_state.new_user_genres = selected_genres
        
        # Show real-time feedback
        if selected_genres:
            st.success(f"✅ Selected genres: {', '.join(selected_genres)}")
            
            # Test genre recommendations immediately
            if st.button("🧪 Test Genre Recommendations", key="test_genre"):
                with st.spinner("Testing genre-based recommendations..."):
                    try:
                        test_recs = get_new_user_recommendations(
                            engine, 
                            None,  # No user ratings
                            selected_genres,
                            5,
                            'genre'
                        )
                        if test_recs:
                            st.write("**Sample recommendations for your genres:**")
                            for i, rec in enumerate(test_recs[:3], 1):
                                title = rec.get('title', 'Unknown')
                                rating = rec.get('predicted_rating', 'N/A')
                                st.write(f"{i}. {title} - {rating:.1f}⭐")
                        else:
                            st.warning("No recommendations found for selected genres.")
                    except Exception as e:
                        st.error(f"Error testing: {e}")
        else:
            st.info("No genres selected yet.")
    
    with tab3:
        st.subheader("🎯 Your Personalized Recommendations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Settings:**")
            
            n_recs = st.slider(
                "Number of recommendations",
                min_value=5,
                max_value=20,
                value=10
            )
            
            # Recommendation method selection
            method_options = {
                "Smart Recommendations": "popularity_genre",
                "Based on Your Ratings": "similarity",
                "Genre-Based": "genre",
                "Most Popular": "popularity"
            }
            
            selected_method_name = st.selectbox(
                "Recommendation Method",
                list(method_options.keys()),
                help="Choose how to generate recommendations"
            )
            method = method_options[selected_method_name]
            
            # Generate recommendations button
            if st.button("🚀 Get My Recommendations", type="primary"):
                with st.spinner("🎬 Generating your personalized recommendations..."):
                    new_user_recommendations = get_new_user_recommendations(
                        engine, 
                        st.session_state.new_user_ratings,
                        st.session_state.new_user_genres,
                        n_recs,
                        method
                    )
                    st.session_state.new_user_recs = new_user_recommendations
                    
                    if new_user_recommendations:
                        st.success("✅ Recommendations generated successfully!")
                    else:
                        st.error("❌ No recommendations could be generated. Please try different settings.")
        
        with col2:
            st.markdown("**Your Recommendations:**")
            
            # Display current selections
            if st.session_state.new_user_ratings or st.session_state.new_user_genres:
                with st.expander("📋 Your Profile Summary"):
                    if st.session_state.new_user_ratings:
                        st.write(f"**Movies Rated:** {len(st.session_state.new_user_ratings)}")
                        avg_rating = np.mean(list(st.session_state.new_user_ratings.values()))
                        st.write(f"**Average Rating:** {avg_rating:.1f} ⭐")
                    
                    if st.session_state.new_user_genres:
                        st.write(f"**Preferred Genres:** {', '.join(st.session_state.new_user_genres)}")
            
            # Display recommendations
            if hasattr(st.session_state, 'new_user_recs') and st.session_state.new_user_recs:
                display_new_user_recommendations(st.session_state.new_user_recs)
            else:
                st.info("Configure your preferences and click 'Get My Recommendations' to see personalized movie suggestions!")

@st.cache_data
def get_popular_movies_for_rating(_ratings_df, _movies_df, n_movies=20):
    """Get popular movies for new users to rate."""
    try:
        # Get movies with most ratings (popular movies new users likely know)
        movie_counts = _ratings_df.groupby('movieId').size().sort_values(ascending=False)
        popular_movie_ids = movie_counts.head(n_movies * 2).index.tolist()
        
        # Filter movies that exist in movies_df
        valid_movie_ids = []
        for movie_id in popular_movie_ids:
            if movie_id in _movies_df['movieId'].values:
                valid_movie_ids.append(movie_id)
                if len(valid_movie_ids) >= n_movies:
                    break
        
        return valid_movie_ids
    except:
        # Fallback to first n movies
        return _movies_df['movieId'].head(n_movies).tolist() if _movies_df is not None else []

def get_new_user_recommendations(_engine, user_ratings, preferred_genres, n_recommendations, method):
    """Get recommendations for a new user without caching (dynamic based on user input)."""
    try:
        if _engine is not None:
            return _engine.get_new_user_recommendations(
                user_preferences=user_ratings if user_ratings else None,
                preferred_genres=preferred_genres if preferred_genres else None,
                n_recommendations=n_recommendations,
                method=method
            )
        else:
            # Fallback recommendations
            return generate_fallback_new_user_recommendations(
                user_ratings, preferred_genres, n_recommendations
            )
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return generate_fallback_new_user_recommendations(
            user_ratings, preferred_genres, n_recommendations
        )

def generate_fallback_new_user_recommendations(user_ratings, preferred_genres, n_recommendations):
    """Generate fallback recommendations for new users."""
    np.random.seed(42)
    
    # Popular movies by genre
    genre_movies = {
        'Action': ['The Dark Knight', 'Mad Max: Fury Road', 'John Wick', 'Die Hard', 'Terminator 2'],
        'Comedy': ['The Hangover', 'Superbad', 'Anchorman', 'Dumb and Dumber', 'Borat'],
        'Drama': ['The Shawshank Redemption', 'Forrest Gump', 'The Godfather', 'Goodfellas', 'Pulp Fiction'],
        'Romance': ['Titanic', 'The Notebook', 'Casablanca', 'When Harry Met Sally', 'Ghost'],
        'Sci-Fi': ['The Matrix', 'Inception', 'Blade Runner', 'Star Wars', 'Interstellar'],
        'Horror': ['The Shining', 'Halloween', 'A Nightmare on Elm Street', 'The Exorcist', 'Get Out'],
        'Animation': ['Toy Story', 'The Lion King', 'Finding Nemo', 'Shrek', 'Up'],
        'Thriller': ['Se7en', 'The Silence of the Lambs', 'North by Northwest', 'Psycho', 'Gone Girl']
    }
    
    recommendations = []
    used_titles = set()
    
    # If user has genre preferences, use those
    if preferred_genres:
        for genre in preferred_genres:
            if genre in genre_movies:
                for title in genre_movies[genre]:
                    if title not in used_titles and len(recommendations) < n_recommendations:
                        recommendations.append({
                            'movieId': len(recommendations) + 1000,
                            'title': title,
                            'predicted_rating': np.random.uniform(4.0, 5.0),
                            'reason': f"Popular {genre} movie matching your preferences",
                            'genres': genre
                        })
                        used_titles.add(title)
    
    # Fill remaining with popular movies
    all_movies = []
    for genre_list in genre_movies.values():
        all_movies.extend(genre_list)
    
    while len(recommendations) < n_recommendations and len(used_titles) < len(all_movies):
        title = np.random.choice([m for m in all_movies if m not in used_titles])
        recommendations.append({
            'movieId': len(recommendations) + 1000,
            'title': title,
            'predicted_rating': np.random.uniform(3.8, 4.8),
            'reason': "Highly rated popular movie",
            'genres': 'Various'
        })
        used_titles.add(title)
    
    return recommendations

def display_new_user_recommendations(recommendations):
    """Display recommendations for new users."""
    if not recommendations:
        st.warning("No recommendations available.")
        return
    
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3 = st.columns([0.5, 3, 1])
            
            with col1:
                st.markdown(f"**#{i}**")
            
            with col2:
                title = rec.get('title', f"Movie ID: {rec.get('movieId', 'Unknown')}")
                st.markdown(f"**{title}**")
                
                reason = rec.get('reason', 'Recommended for you')
                st.caption(f"💡 {reason}")
                
                if 'genres' in rec or 'matching_genres' in rec:
                    genres = rec.get('matching_genres', rec.get('genres', ''))
                    if genres:
                        if isinstance(genres, list):
                            genres_str = ', '.join(genres)
                        else:
                            genres_str = str(genres)
                        st.caption(f"🎭 {genres_str}")
            
            with col3:
                rating = rec.get('predicted_rating', 0)
                st.metric("Rating", f"{rating:.1f} ⭐")
                
                if 'rating_count' in rec:
                    st.caption(f"{rec['rating_count']} ratings")
            
            st.divider()

def main():
    """Main Streamlit application."""
    display_main_header()
    
    # Load data and engine
    with st.spinner("🔄 Loading recommendation system..."):
        ratings_df, movies_df, is_real_data = load_sample_data()
        engine, engine_loaded = load_recommendation_engine()
    
    if ratings_df is None:
        st.error("❌ Failed to load data. Please check your data files.")
        return
    
    # Sidebar information
    display_sidebar_info(ratings_df, movies_df, is_real_data)
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "🎯 Recommendations", 
        "👤 User Profile", 
        "🆕 New User", 
        "📈 Analytics"
    ])
    
    with tab1:
        display_system_overview(ratings_df, movies_df)
    
    with tab2:
        display_recommendation_interface(engine, ratings_df, movies_df)
    
    with tab3:
        available_users = sorted(ratings_df['userId'].unique())
        selected_user_profile = st.selectbox(
            "Select User for Profile Analysis",
            available_users,
            index=0
        )
        display_user_profile(ratings_df, movies_df, selected_user_profile)
    
    with tab4:
        display_new_user_interface(engine, ratings_df, movies_df)
    
    with tab5:
        display_analytics_dashboard(ratings_df, movies_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            🎬 Movie Recommendation System | Built with Streamlit & Python<br>
            Combining Collaborative Filtering + Machine Learning for Personalized Recommendations
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
