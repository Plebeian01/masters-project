import pandas as pd
import numpy as np
import os
from surprise import Dataset, Reader, SVD
from collections import defaultdict
import xgboost as xgb
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import logging
import shap
import lime
import lime.lime_tabular
import random
import openai

logger = logging.getLogger(__name__)

class MovieRecommendationModel:
    def __init__(self):
        self.model = None
        self.user_factors_df = None
        self.movie_features_df = None
        self.title_to_movie_id = None
        self.movie_id_to_title = None
        self.ratings_df = None
        self.one_hot_genre_columns = None
        self.original_X_columns = None
        self.N_FACTORS_SVD = 10
        self.median_year = None
        self.global_avg_movie_rating = None
        self.shap_background = None  # Store background for SHAP
        self.lime_explainer = None   # Store LIME explainer
        self.is_trained = False
        
    def load_and_train_model(self, data_dir="data/100kDataset"):
        """Load data and train the recommendation model"""
        try:
            logger.info("Loading data...")
            
            # Load data
            movies_csv_path = os.path.join(data_dir, 'movies.csv')
            ratings_csv_path = os.path.join(data_dir, 'ratings.csv')
            
            if not os.path.exists(movies_csv_path) or not os.path.exists(ratings_csv_path):
                raise FileNotFoundError(f"Data files not found in {data_dir}")
            
            movies_df_orig = pd.read_csv(movies_csv_path)
            ratings_df_orig = pd.read_csv(ratings_csv_path)
            
            # Create copies
            movies_df = movies_df_orig.copy()
            ratings_df = ratings_df_orig.copy()
            
            # Remove movies with less than 3 ratings
            movie_rating_counts = ratings_df.groupby('movieId')['rating'].count().rename('movie_num_ratings').reset_index()
            cutoff_list = movie_rating_counts.loc[movie_rating_counts['movie_num_ratings'] < 3, 'movieId'].tolist()
            
            movies_df = movies_df[~movies_df['movieId'].isin(cutoff_list)].reset_index(drop=True)
            ratings_df = ratings_df[~ratings_df['movieId'].isin(cutoff_list)].reset_index(drop=True)
            
            # Calculate genre averages
            movie_ratings_for_genre_avg = pd.merge(movies_df, ratings_df, on='movieId')
            temp_genre_df = movie_ratings_for_genre_avg[['movieId', 'genres', 'rating']].copy()
            temp_genre_df['genres_list'] = temp_genre_df['genres'].str.split('|')
            exploded_genres_ratings = temp_genre_df.explode('genres_list')
            genre_avg_ratings_series = exploded_genres_ratings.groupby('genres_list')['rating'].mean()
            genre_to_avg_rating_map = genre_avg_ratings_series.to_dict()
            
            # Global average
            self.global_avg_movie_rating = ratings_df['rating'].mean()
            
            # Movie feature engineering
            logger.info("Engineering movie features...")
            movies_df['num_genres'] = movies_df['genres'].apply(lambda x: len(x.split('|')) if x != '(no genres listed)' else 0)
            movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
            movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
            self.median_year = movies_df['year'].median()
            movies_df['year'] = movies_df['year'].fillna(self.median_year)
            
            movies_df['is_old_movie'] = (movies_df['year'] < 1990).astype(int)
            movies_df['is_recent_movie'] = (movies_df['year'] > 2015).astype(int)
            
            # Decade features
            movies_df['decade'] = (movies_df['year'] // 10) * 10
            movies_df['decade'] = movies_df['decade'].astype(int)
            decade_dummies = pd.get_dummies(movies_df['decade'], prefix='decade', dummy_na=False, dtype=int)
            movies_df = pd.concat([movies_df, decade_dummies], axis=1)
            one_hot_decade_columns = decade_dummies.columns.tolist()
            
            # Movie stats
            avg_series = ratings_df.groupby('movieId')['rating'].mean()
            count_series = ratings_df.groupby('movieId')['rating'].count()
            movies_df['avg_rating'] = movies_df['movieId'].map(avg_series).fillna(self.global_avg_movie_rating)
            movies_df['num_ratings'] = movies_df['movieId'].map(count_series).fillna(0).astype(int)
            
            # Genre popularity
            def calculate_movie_genre_avg_pop(genres_str, genre_map, default_val):
                if genres_str == '(no genres listed)' or pd.isna(genres_str):
                    return default_val
                movie_genres = genres_str.split('|')
                pop_sum = 0
                count = 0
                for genre in movie_genres:
                    if genre in genre_map:
                        pop_sum += genre_map[genre]
                        count += 1
                return pop_sum / count if count > 0 else default_val
            
            movies_df['movie_genre_avg_popularity'] = movies_df['genres'].apply(
                lambda x: calculate_movie_genre_avg_pop(x, genre_to_avg_rating_map, self.global_avg_movie_rating)
            )
            
            # One-hot encode genres
            genres_dummies_movies = movies_df['genres'].str.get_dummies(sep='|')
            genres_dummies_movies.columns = [f"genre_{col.replace(' ', '_').replace('-', '_')}" for col in genres_dummies_movies.columns]
            self.one_hot_genre_columns = genres_dummies_movies.columns.tolist()
            movies_df = pd.concat([movies_df, genres_dummies_movies], axis=1)
            
            # SVD training
            logger.info("Training SVD model...")
            df = pd.merge(ratings_df, movies_df[['movieId', 'title']], on='movieId')
            reader = Reader(rating_scale=(0.5, 5.0))
            surprise_data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
            
            svd_model = SVD(n_factors=self.N_FACTORS_SVD, n_epochs=20, random_state=42, verbose=False)
            full_trainset_svd = surprise_data.build_full_trainset()
            svd_model.fit(full_trainset_svd)
            
            # Extract user factors
            user_factors_list = []
            for inner_uid in full_trainset_svd.all_users():
                raw_uid = full_trainset_svd.to_raw_uid(inner_uid)
                factors = svd_model.pu[inner_uid]
                user_factors_list.append([raw_uid] + factors.tolist())
            self.user_factors_df = pd.DataFrame(user_factors_list, columns=['userId'] + [f'uf_svd_{i}' for i in range(self.N_FACTORS_SVD)])
            self.user_factors_df = self.user_factors_df.set_index('userId')
            
            # Extract item factors
            item_factors_list = []
            for inner_iid in full_trainset_svd.all_items():
                raw_iid = full_trainset_svd.to_raw_iid(inner_iid)
                factors = svd_model.qi[inner_iid]
                item_factors_list.append([raw_iid] + factors.tolist())
            item_factors_df = pd.DataFrame(item_factors_list, columns=['movieId'] + [f'if_svd_{i}' for i in range(self.N_FACTORS_SVD)])
            item_factors_df = item_factors_df.set_index('movieId')
            
            # User features
            logger.info("Engineering user features...")
            user_stats = ratings_df.groupby('userId')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'user_avg_rating', 'count': 'user_num_ratings'})
            user_movie_years = pd.merge(ratings_df, movies_df[['movieId', 'year']], on='movieId')
            user_newness_pref_series = user_movie_years.groupby('userId')['year'].mean().rename('user_newness_pref')
            
            # User genre diversity
            user_genre_ratings = pd.merge(ratings_df[['userId', 'movieId']], movies_df[['movieId'] + self.one_hot_genre_columns], on='movieId')
            user_genre_counts = user_genre_ratings.groupby('userId')[self.one_hot_genre_columns].sum()
            
            def calculate_genre_entropy(row):
                genre_counts_for_user = row[row > 0]
                if genre_counts_for_user.empty:
                    return 0
                probabilities = genre_counts_for_user / genre_counts_for_user.sum()
                return entropy(probabilities, base=2)
            
            user_genre_diversity_series = user_genre_counts.apply(calculate_genre_entropy, axis=1).rename('user_genre_diversity')
            
            # Combine user features
            user_features_df = self.user_factors_df.merge(user_stats, on='userId', how='left')
            user_features_df = user_features_df.merge(user_newness_pref_series, on='userId', how='left')
            user_features_df = user_features_df.merge(user_genre_diversity_series, on='userId', how='left')
            user_features_df = user_features_df.fillna({
                'user_avg_rating': full_trainset_svd.global_mean,
                'user_num_ratings': 0,
                'user_newness_pref': self.median_year,
                'user_genre_diversity': 0
            })
            
            # Movie features
            logger.info("Finalizing movie features...")
            movie_meta_features_to_select = ['movieId', 'num_genres', 'year', 'is_old_movie', 'is_recent_movie', 'movie_genre_avg_popularity'] + one_hot_decade_columns + self.one_hot_genre_columns
            base_movie_meta_features_df = movies_df[movie_meta_features_to_select].set_index('movieId')
            
            movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'movie_avg_rating', 'count': 'movie_num_ratings'})
            self.movie_features_df = item_factors_df.merge(movie_stats, on='movieId', how='left')
            self.movie_features_df = self.movie_features_df.merge(base_movie_meta_features_df, on='movieId', how='left')
            
            # Fill NaNs
            self.movie_features_df['movie_avg_rating'] = self.movie_features_df['movie_avg_rating'].fillna(full_trainset_svd.global_mean)
            self.movie_features_df['movie_num_ratings'] = self.movie_features_df['movie_num_ratings'].fillna(0)
            self.movie_features_df['year'] = self.movie_features_df['year'].fillna(self.median_year)
            self.movie_features_df['num_genres'] = self.movie_features_df['num_genres'].fillna(0)
            self.movie_features_df['is_old_movie'] = self.movie_features_df['is_old_movie'].fillna(False)
            self.movie_features_df['is_recent_movie'] = self.movie_features_df['is_recent_movie'].fillna(False)
            self.movie_features_df['movie_genre_avg_popularity'] = self.movie_features_df['movie_genre_avg_popularity'].fillna(self.global_avg_movie_rating)
            
            for col in one_hot_decade_columns:
                self.movie_features_df[col] = self.movie_features_df[col].fillna(0)
            for col in self.one_hot_genre_columns:
                self.movie_features_df[col] = self.movie_features_df[col].fillna(0)
            
            # Train XGBoost
            logger.info("Training XGBoost model...")
            xgb_train_df = ratings_df_orig.merge(user_features_df, on='userId', how='left')
            xgb_train_df = xgb_train_df.merge(self.movie_features_df, on='movieId', how='left')
            
            y = xgb_train_df['rating']
            columns_to_drop_for_X = ['userId', 'movieId', 'rating', 'timestamp']
            if 'title' in xgb_train_df.columns:
                columns_to_drop_for_X.append('title')
            if 'genres' in xgb_train_df.columns:
                columns_to_drop_for_X.append('genres')
            
            X = xgb_train_df.drop(columns=columns_to_drop_for_X, errors='ignore')
            X = X.fillna(0)
            X.columns = [str(col) for col in X.columns]
            self.original_X_columns = X.columns.tolist()
            
            X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Store a sample of training data for SHAP background
            self.shap_background = X_train.sample(n=min(100, len(X_train)), random_state=42)
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                mode='regression',
                random_state=42
            )
            
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100, learning_rate=0.1, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
            # Store mappings
            self.title_to_movie_id = {title: id for id, title in movies_df_orig.set_index('movieId')['title'].to_dict().items()}
            self.movie_id_to_title = movies_df_orig.set_index('movieId')['title'].to_dict()
            self.ratings_df = ratings_df
            
            # Test performance
            y_pred_test = self.model.predict(X_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            logger.info(f"XGBoost Test RMSE: {rmse_test:.4f}")
            
            self.is_trained = True
            logger.info("Model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise e
    
    def get_recommendations(self, user_ratings_input, n=5):
        """Get recommendations for a new user"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        liked_movie_ids = set()
        valid_ratings = []
        
        for title, rating in user_ratings_input:
            mid = self.title_to_movie_id.get(title)
            if mid:
                liked_movie_ids.add(mid)
                valid_ratings.append(rating)
            else:
                logger.warning(f"Movie '{title}' not found in database")
        
        if not liked_movie_ids:
            return []
        
        user_avg = np.mean(valid_ratings) if valid_ratings else 0.0
        user_count = len(valid_ratings)
        
        # Find similar users
        sim_users = self.find_similar_users(liked_movie_ids, self.ratings_df, 4.0)
        
        if len(sim_users) < 5:
            est_factors = self.user_factors_df.mean()
        else:
            valid_sim_users = self.user_factors_df.index.intersection(sim_users)
            if not valid_sim_users.empty:
                sim_df = self.user_factors_df.loc[valid_sim_users]
                est_factors = sim_df.mean()
            else:
                est_factors = self.user_factors_df.mean()
        
        # Build genre profile
        def make_genre_profile(lm_ids, mf_df, gcols):
            valid_lm_ids = mf_df.index.intersection(lm_ids)
            if not valid_lm_ids.empty:
                sub = mf_df.loc[valid_lm_ids, gcols]
                return sub.mean().values.reshape(1, -1)
            else:
                return np.zeros((1, len(gcols)))
        
        user_genre_vec = make_genre_profile(liked_movie_ids, self.movie_features_df, self.one_hot_genre_columns)
        
        # Assemble user features
        user_feats = {
            'user_avg_rating': user_avg,
            'user_num_ratings': user_count,
            'user_newness_pref': self.median_year,
            'user_genre_diversity': 0
        }
        
        for i, factor_val in enumerate(est_factors):
            user_feats[f'uf_svd_{i}'] = factor_val
        
        # Score candidates
        cands = []
        for mid in self.movie_features_df.index:
            if mid in liked_movie_ids:
                continue
            
            mf = self.movie_features_df.loc[mid]
            if mf.get('movie_num_ratings', 0) < 10:
                continue
            
            # Genre similarity
            mv = mf[self.one_hot_genre_columns].values.reshape(1, -1)
            sim = float(cosine_similarity(user_genre_vec, mv)[0, 0])
            if sim < 0.1:
                continue
            
            # Build feature vector
            row = {
                col: user_feats.get(col, mf.get(col, 0))
                for col in self.original_X_columns
            }
            fv = pd.DataFrame([row], columns=self.original_X_columns).fillna(0)
            
            pr = self.model.predict(fv)[0]
            if pr < 4.0:
                continue
            
            avg = mf.get('movie_avg_rating', 3.5)
            uplift = pr - avg
            adj = uplift * sim
            
            cands.append({
                'movieId': mid,
                'predicted_rating': pr,
                'uplift': uplift,
                'genre_similarity': sim,
                'adjusted_uplift': adj
            })
        
        # Return top-N by adjusted uplift
        cands.sort(key=lambda x: x['adjusted_uplift'], reverse=True)
        return [
            {
                'title': self.movie_id_to_title.get(x['movieId'], 'Unknown'),
                'predicted_rating': float(x['predicted_rating']),
                'uplift': float(x['uplift']),
                'genre_similarity': float(x['genre_similarity']),
                'adjusted_uplift': float(x['adjusted_uplift']),
                'movieId': int(x['movieId'])
            }
            for x in cands[:n]
        ]
    
    def find_similar_users(self, liked_movie_ids, ratings_df, thresh=4.0):
        """Helper function to find similar users"""
        users = set()
        for m_id in liked_movie_ids:
            dfm = ratings_df[(ratings_df.movieId == m_id) & (ratings_df.rating >= thresh)]
            users.update(dfm.userId.tolist())
        return list(users)
    
    def get_recommendations_for_study(self, user_ratings_input, n=5):
        """
        Get recommendations with randomized explanation types for user study,
        ensuring that each explanation type appears exactly once per batch.
        """
    
        recommendations = self.get_recommendations(user_ratings_input, n + 5)
        if not recommendations:
            return []
    
        top_recommendations = recommendations[:n]
    
        # Insert decoy/baseline
        baseline_movie = self._select_baseline_movie(recommendations, top_recommendations)
        if baseline_movie:
            replace_idx = random.randint(0, len(top_recommendations) - 1)
            top_recommendations[replace_idx] = baseline_movie
    
        # Assign explanation types
        explanation_types = ['none', 'basic', 'shap', 'lime', 'llm']
        random.shuffle(explanation_types)  # Randomize order for each batch
    
        for rec, explanation_type in zip(top_recommendations, explanation_types):
            rec['is_baseline'] = rec.get('is_baseline', False)
            rec['explanation_type'] = explanation_type
            rec['explanation'] = self.generate_explanation(rec, user_ratings_input, explanation_type)
    
        return top_recommendations
            
    def _select_baseline_movie(self, all_recommendations, top_recommendations):
        """
        Select a decoy (baseline) movie that feels like a generic, plausible recommendation:
        - Moderate popularity and average global rating.
        - Weakly similar to the user's preferences (not completely irrelevant).
        - Always returns one decoy (falls back to random non-top rec if needed).
        - Displayed predicted rating is faked to look convincing.
        """
        try:
            top_ids = {rec['movieId'] for rec in top_recommendations}
    
            # ---- ✅ Build user genre profile ----
            liked_movie_ids = [rec['movieId'] for rec in top_recommendations]
            user_genre_vec = np.zeros((1, len(self.one_hot_genre_columns)))
    
            if liked_movie_ids:
                valid_liked_ids = self.movie_features_df.index.intersection(liked_movie_ids)
                if not valid_liked_ids.empty:
                    user_genre_vec = self.movie_features_df.loc[
                        valid_liked_ids, self.one_hot_genre_columns
                    ].mean().values.reshape(1, -1)
    
            # ---- ✅ Select "middle-of-the-road" candidates ----
            candidates = []
            for mid, mf in self.movie_features_df.iterrows():
                if mid in top_ids:
                    continue
    
                # Compute similarity to user's preferences
                sim = float(
                    cosine_similarity(
                        user_genre_vec, mf[self.one_hot_genre_columns].values.reshape(1, -1)
                    )[0, 0]
                )
    
                # Get global stats (default to average-like values if missing)
                global_avg = mf.get("global_avg_rating", 3.5)
                num_ratings = mf.get("movie_num_ratings", 100)
    
                # Criteria for plausible "average" recommendations
                if (
                    0.05 <= sim <= 0.15  # weak but not zero similarity
                    and 3.0 <= global_avg <= 3.7
                    and 50 <= num_ratings <= 5000
                ):
                    candidates.append({
                        "movieId": int(mid),
                        "title": self.movie_id_to_title.get(mid, "Unknown"),
                        "predicted_rating": global_avg,
                        "uplift": 0.0,
                        "genre_similarity": sim,
                        "adjusted_uplift": 0.0
                    })
    
            # ---- ✅ Sort by "middle-of-the-roadness" ----
            if candidates:
                candidates.sort(key=lambda x: abs(x["predicted_rating"] - 3.5))
                baseline = random.choice(candidates[:10]) if len(candidates) > 10 else candidates[0]
            else:
                # Fallback: pick any non-top recommendation
                fallback_candidates = [rec for rec in all_recommendations if rec['movieId'] not in top_ids]
                if not fallback_candidates:
                    return None
                baseline = random.choice(fallback_candidates)
    
            # ---- ✅ Make it LOOK like a strong recommendation ----
            baseline_display = baseline.copy()
            baseline_display["predicted_rating"] = round(random.uniform(4.3, 4.6), 3)
            baseline_display["is_baseline"] = True
    
            logger.info(
                f"Selected baseline decoy: {baseline_display['title']} "
                f"(real avg_rating={baseline.get('predicted_rating', 3.5):.2f}, "
                f"genre_sim={baseline['genre_similarity']:.3f})"
            )
    
            return baseline_display
    
        except Exception as e:
            logger.warning(f"Could not select baseline movie: {str(e)}")
            return None
    
    def generate_explanation(self, recommendation, user_ratings_input, explanation_type):
        """Generate explanation based on type"""
        if explanation_type == 'none':
            return None
        elif explanation_type == 'basic':
            return {
                'type': 'basic',
                'content': f"Predicted rating: {recommendation['predicted_rating']:.3f}/5, Uplift score: {recommendation['uplift']:.3f}"
            }
        elif explanation_type == 'shap':
            return self.generate_shap_explanation(recommendation, user_ratings_input)
        elif explanation_type == 'lime':
            return self.generate_lime_explanation(recommendation, user_ratings_input)
        elif explanation_type == 'llm':
            return self.generate_llm_explanation(recommendation, user_ratings_input)
        
        return None
    
    def generate_shap_explanation(self, recommendation, user_ratings_input):
        """Generate SHAP explanation"""
        try:
            movieId = recommendation['movieId']
            
            # Build feature vector (same logic as before)
            liked_movie_ids = set()
            valid_ratings = []
            for title, rating in user_ratings_input:
                mid = self.title_to_movie_id.get(title)
                if mid:
                    liked_movie_ids.add(mid)
                    valid_ratings.append(rating)
            
            user_avg = np.mean(valid_ratings) if valid_ratings else 0.0
            user_count = len(valid_ratings)
            
            # Get similar users for SVD factors
            sim_users = self.find_similar_users(liked_movie_ids, self.ratings_df, 4.0)
            
            if len(sim_users) < 5:
                est_factors = self.user_factors_df.mean()
            else:
                valid_sim_users = self.user_factors_df.index.intersection(sim_users)
                if not valid_sim_users.empty:
                    sim_df = self.user_factors_df.loc[valid_sim_users]
                    est_factors = sim_df.mean()
                else:
                    est_factors = self.user_factors_df.mean()
            
            # Assemble user features
            user_feats = {
                'user_avg_rating': user_avg,
                'user_num_ratings': user_count,
                'user_newness_pref': self.median_year,
                'user_genre_diversity': 0
            }
            
            for i, factor_val in enumerate(est_factors):
                user_feats[f'uf_svd_{i}'] = factor_val
            
            # Get movie features
            if movieId in self.movie_features_df.index:
                mf = self.movie_features_df.loc[movieId]
                
                # Build feature vector
                row = {
                    col: user_feats.get(col, mf.get(col, 0))
                    for col in self.original_X_columns
                }
                fv = pd.DataFrame([row], columns=self.original_X_columns).fillna(0)
                
                # Generate SHAP explanation
                if self.shap_background is not None:
                    explainer = shap.TreeExplainer(self.model, data=self.shap_background)
                else:
                    explainer = shap.TreeExplainer(self.model, data=fv)
                
                shap_values = explainer.shap_values(fv)
                
                # Get top contributing features
                feature_contributions = list(zip(fv.columns, shap_values[0]))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Filter significant features
                significant_features = [
                    {
                        'feature': feat,
                        'contribution': float(contrib),
                        'feature_value': float(fv[feat].iloc[0])
                    }
                    for feat, contrib in feature_contributions[:10]
                    if abs(contrib) > 1e-6
                ][:5]
                
                return {
                    'type': 'shap',
                    'baseline': float(explainer.expected_value),
                    'features': significant_features
                }
        
        except Exception as e:
            logger.warning(f"Could not generate SHAP explanation: {str(e)}")
            return None
    
    def generate_lime_explanation(self, recommendation, user_ratings_input):
        """Generate LIME explanation"""
        try:
            movieId = recommendation['movieId']
            
            # Build feature vector (same logic as SHAP)
            liked_movie_ids = set()
            valid_ratings = []
            for title, rating in user_ratings_input:
                mid = self.title_to_movie_id.get(title)
                if mid:
                    liked_movie_ids.add(mid)
                    valid_ratings.append(rating)
            
            user_avg = np.mean(valid_ratings) if valid_ratings else 0.0
            user_count = len(valid_ratings)
            
            sim_users = self.find_similar_users(liked_movie_ids, self.ratings_df, 4.0)
            
            if len(sim_users) < 5:
                est_factors = self.user_factors_df.mean()
            else:
                valid_sim_users = self.user_factors_df.index.intersection(sim_users)
                if not valid_sim_users.empty:
                    sim_df = self.user_factors_df.loc[valid_sim_users]
                    est_factors = sim_df.mean()
                else:
                    est_factors = self.user_factors_df.mean()
            
            user_feats = {
                'user_avg_rating': user_avg,
                'user_num_ratings': user_count,
                'user_newness_pref': self.median_year,
                'user_genre_diversity': 0
            }
            
            for i, factor_val in enumerate(est_factors):
                user_feats[f'uf_svd_{i}'] = factor_val
            
            if movieId in self.movie_features_df.index:
                mf = self.movie_features_df.loc[movieId]
                
                row = {
                    col: user_feats.get(col, mf.get(col, 0))
                    for col in self.original_X_columns
                }
                fv = pd.DataFrame([row], columns=self.original_X_columns).fillna(0)
                
                # Generate LIME explanation
                if self.lime_explainer is not None:
                    instance = fv.iloc[0].values
                    
                    def predict_fn(X):
                        df = pd.DataFrame(X, columns=self.original_X_columns)
                        return self.model.predict(df)
                    
                    explanation = self.lime_explainer.explain_instance(
                        instance, predict_fn, num_features=5
                    )
                    
                    # Extract feature contributions
                    lime_features = []
                    for feature_name, contribution in explanation.as_list():
                        lime_features.append({
                            'feature': feature_name,
                            'contribution': float(contribution)
                        })
                    
                    return {
                        'type': 'lime',
                        'features': lime_features
                    }
        
        except Exception as e:
            logger.warning(f"Could not generate LIME explanation: {str(e)}")
            return None
    
    def generate_llm_explanation(self, recommendation, user_ratings_input=None):
        """Generate LLM-style explanation using OpenAI"""
        
        # Get OpenAI API key from environment variable
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("OpenAI API key not found, using placeholder explanation")
            return self._get_placeholder_explanation(recommendation)
        
        try:
            # Set up OpenAI client
            client = openai.OpenAI(api_key=api_key)
            
            # Build context about user's preferences
            user_context = ""
            if user_ratings_input:
                liked_movies = [title for title, rating in user_ratings_input if rating >= 4]
                if liked_movies:
                    user_context = f"The user has rated these movies highly: {', '.join(liked_movies[:3])}. "
            
            # Create a focused prompt for movie recommendation explanation
            prompt = f"""You are explaining why a movie recommendation system suggested a specific movie to a user. 

{user_context}Based on this, the system recommended: "{recommendation['title']}"

Write a brief, personalized explanation (2-3 sentences) for why this movie was recommended. Focus on:
- How it relates to their viewing preferences
- What makes this movie a good match
- Maintaining a polite and professional tone throughout your response
- Keeping responses brief and to-the-point, without any fluff text

All recommendations make sense as a movie the user is expected to like based on their preferences.

Explanation:"""

            # Call OpenAI API
            logger.debug(f"Sending prompt to OpenAI: {prompt}")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful movie recommendation explainer. Keep explanations brief and personalized."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            explanation_text = response.choices[0].message.content.strip()
            
            logger.info(f"Generated OpenAI explanation for {recommendation['title']}")
            
            return {
                'type': 'llm',
                'content': explanation_text,
                'source': 'openai'
            }
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            # Fallback to placeholder
            return self._get_placeholder_explanation(recommendation)
    
    def _get_placeholder_explanation(self, recommendation):
        """Fallback placeholder explanations"""
        explanations = [
            f"Based on your viewing history, we think you'll enjoy '{recommendation['title']}' because it shares similar themes and style with movies you've rated highly.",
            f"'{recommendation['title']}' is recommended for you due to its strong ratings from users with similar preferences to yours.",
            f"Our analysis suggests '{recommendation['title']}' matches your taste profile, particularly in terms of genre preferences and rating patterns.",
            f"You might like '{recommendation['title']}' as it's popular among users who share your movie preferences and rating behavior.",
            f"'{recommendation['title']}' appears in your recommendations because it aligns with the characteristics of movies you typically enjoy."
        ]
        
        return {
            'type': 'llm',
            'content': random.choice(explanations),
            'source': 'placeholder'
        }
    
    def get_available_movies(self, search_term=""):
        """Get list of available movies for frontend autocomplete"""
        if not self.title_to_movie_id:
            return []
        
        movies = list(self.title_to_movie_id.keys())
        
        if search_term:
            search_term = search_term.lower()
            movies = [m for m in movies if search_term in m.lower()]
        
        return sorted(movies)[:100]  # Limit to 100 results
    
    def get_recommendations_with_explanations(self, user_ratings_input, n=5, include_explanations=True):
        """Get recommendations with SHAP/LIME explanations"""
        # Get base recommendations using existing method
        recommendations = self.get_recommendations(user_ratings_input, n)
        
        if not include_explanations or not recommendations:
            return recommendations
        
        try:
            # Generate explanations for each recommendation
            for rec in recommendations:
                movieId = rec['movieId']
                
                # Create a dummy user feature vector for explanation
                liked_movie_ids = set()
                valid_ratings = []
                for title, rating in user_ratings_input:
                    mid = self.title_to_movie_id.get(title)
                    if mid:
                        liked_movie_ids.add(mid)
                        valid_ratings.append(rating)
                
                user_avg = np.mean(valid_ratings) if valid_ratings else 0.0
                user_count = len(valid_ratings)
                
                # Find similar users for SVD factors
                sim_users = self.find_similar_users(liked_movie_ids, self.ratings_df, 4.0)
                
                if len(sim_users) < 5:
                    est_factors = self.user_factors_df.mean()
                else:
                    valid_sim_users = self.user_factors_df.index.intersection(sim_users)
                    if not valid_sim_users.empty:
                        sim_df = self.user_factors_df.loc[valid_sim_users]
                        est_factors = sim_df.mean()
                    else:
                        est_factors = self.user_factors_df.mean()
                
                # Assemble user features
                user_feats = {
                    'user_avg_rating': user_avg,
                    'user_num_ratings': user_count,
                    'user_newness_pref': self.median_year,
                    'user_genre_diversity': 0
                }
                
                for i, factor_val in enumerate(est_factors):
                    user_feats[f'uf_svd_{i}'] = factor_val
                
                # Get movie features
                if movieId in self.movie_features_df.index:
                    mf = self.movie_features_df.loc[movieId]
                    
                    # Build feature vector for explanation
                    row = {
                        col: user_feats.get(col, mf.get(col, 0))
                        for col in self.original_X_columns
                    }
                    fv = pd.DataFrame([row], columns=self.original_X_columns).fillna(0)
                    
                    # Generate SHAP explanation
                    try:
                        # Use stored background data for better SHAP explanations
                        if self.shap_background is not None:
                            explainer = shap.TreeExplainer(self.model, data=self.shap_background)
                        else:
                            # Fallback to using the single feature vector
                            explainer = shap.TreeExplainer(self.model, data=fv)
                        
                        shap_values = explainer.shap_values(fv)
                        
                        # Get top contributing features
                        feature_contributions = list(zip(fv.columns, shap_values[0]))
                        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                        
                        # Only include features with non-zero contributions
                        significant_features = [
                            {
                                'feature': feat,
                                'contribution': float(contrib),
                                'feature_value': float(fv[feat].iloc[0])
                            }
                            for feat, contrib in feature_contributions[:10]  # Take top 10, filter below
                            if abs(contrib) > 1e-6  # Only show contributions > 0.000001
                        ][:5]  # Then take top 5
                        
                        rec['explanations'] = {
                            'shap_baseline': float(explainer.expected_value),
                            'top_features': significant_features
                        }
                    except Exception as e:
                        logger.warning(f"Could not generate SHAP explanation for {movieId}: {str(e)}")
                        rec['explanations'] = None
                
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            # Return recommendations without explanations if explanation fails
            pass
        
        return recommendations