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
from sklearn.linear_model import Ridge
import logging
import shap
import lime
import lime.lime_tabular as llt
import random
import openai
import re

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
        self.X_train_ref = None      # Keep X_train for LIME tuning and diagnostics
        self.shap_explainer = None   # Cache SHAP TreeExplainer


    def load_and_train_model(self, data_dir="data/100kDataset"):
        """Load data and train the recommendation model (SVD features + XGBoost)."""
        try:
            logger.info("Loading data.")
            movies_csv_path = os.path.join(data_dir, 'movies.csv')
            ratings_csv_path = os.path.join(data_dir, 'ratings.csv')
            if not os.path.exists(movies_csv_path) or not os.path.exists(ratings_csv_path):
                raise FileNotFoundError(f"Data files not found in {data_dir}")
    
            movies_df_orig = pd.read_csv(movies_csv_path)
            ratings_df_orig = pd.read_csv(ratings_csv_path)
    
            # Work on copies
            movies_df = movies_df_orig.copy()
            ratings_df = ratings_df_orig.copy()
    
            # Filtering & globals 
            # Drop extremely sparse movies (<5)
            movie_rating_counts = (
                ratings_df.groupby('movieId')['rating']
                .count().rename('movie_num_ratings').reset_index()
            )
            cutoff_list = movie_rating_counts.loc[movie_rating_counts['movie_num_ratings'] < 5, 'movieId'].tolist()
            movies_df = movies_df[~movies_df['movieId'].isin(cutoff_list)].reset_index(drop=True)
            ratings_df = ratings_df[~ratings_df['movieId'].isin(cutoff_list)].reset_index(drop=True)
    
            # Global average rating
            self.global_avg_movie_rating = ratings_df['rating'].mean()
    
            # Movie feature engineering
            logger.info("Engineering movie features.")
            movies_df['num_genres'] = movies_df['genres'].apply(
                lambda x: len(x.split('|')) if x != '(no genres listed)' else 0
            )
            movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
            movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
            self.median_year = movies_df['year'].median()
            movies_df['year'] = movies_df['year'].fillna(self.median_year)
    
            movies_df['is_old_movie'] = (movies_df['year'] < 1990).astype(int)
            movies_df['is_recent_movie'] = (movies_df['year'] > 2015).astype(int)
    
            # Decade one-hots
            movies_df['decade'] = (movies_df['year'] // 10) * 10
            movies_df['decade'] = movies_df['decade'].astype(int)
            decade_dummies = pd.get_dummies(movies_df['decade'], prefix='decade', dummy_na=False, dtype=int)
            movies_df = pd.concat([movies_df, decade_dummies], axis=1)
            one_hot_decade_columns = decade_dummies.columns.tolist()  # keep for later merges  
    
            # Movie stats
            avg_series = ratings_df.groupby('movieId')['rating'].mean()
            count_series = ratings_df.groupby('movieId')['rating'].count()
            movies_df['avg_rating'] = movies_df['movieId'].map(avg_series).fillna(self.global_avg_movie_rating)
            movies_df['num_ratings'] = movies_df['movieId'].map(count_series).fillna(0).astype(int)
    
            # Genre popularity (average rating per genre)
            movie_ratings_for_genre_avg = pd.merge(movies_df, ratings_df, on='movieId')
            temp_genre_df = movie_ratings_for_genre_avg[['movieId', 'genres', 'rating']].copy()
            temp_genre_df['genres_list'] = temp_genre_df['genres'].str.split('|')
            exploded_genres_ratings = temp_genre_df.explode('genres_list')
            genre_avg_ratings_series = exploded_genres_ratings.groupby('genres_list')['rating'].mean()
            genre_to_avg_rating_map = genre_avg_ratings_series.to_dict()
    
            def calculate_movie_genre_avg_pop(genres_str, genre_map, default_val):
                if genres_str == '(no genres listed)' or pd.isna(genres_str):
                    return default_val
                vals = []
                for g in str(genres_str).split('|'):
                    if g in genre_map:
                        vals.append(genre_map[g])
                return np.mean(vals) if len(vals) else default_val
    
            movies_df['movie_genre_avg_popularity'] = movies_df['genres'].apply(
                lambda x: calculate_movie_genre_avg_pop(x, genre_to_avg_rating_map, self.global_avg_movie_rating)
            )  
    
            # Genre one-hots 
            genres_dummies_movies = movies_df['genres'].str.get_dummies(sep='|')
            genres_dummies_movies.columns = [
                f"genre_{col.replace(' ', '_').replace('-', '_')}" for col in genres_dummies_movies.columns
            ]
            self.one_hot_genre_columns = genres_dummies_movies.columns.tolist()
            movies_df = pd.concat([movies_df, genres_dummies_movies], axis=1)  
    
            # Surprise SVD Factors
            logger.info("Training SVD model.")
            df_svd = pd.merge(ratings_df, movies_df[['movieId', 'title']], on='movieId')
            reader = Reader(rating_scale=(0.5, 5.0))
            surprise_data = Dataset.load_from_df(df_svd[['userId', 'movieId', 'rating']], reader)
            svd_model = SVD(n_factors=self.N_FACTORS_SVD, n_epochs=20, random_state=42, verbose=False)
            full_trainset_svd = surprise_data.build_full_trainset()
            svd_model.fit(full_trainset_svd)
    
            # Extract user factors
            user_factors_list = []
            for inner_uid in full_trainset_svd.all_users():
                raw_uid = full_trainset_svd.to_raw_uid(inner_uid)
                factors = svd_model.pu[inner_uid]
                user_factors_list.append([raw_uid] + factors.tolist())
            self.user_factors_df = pd.DataFrame(
                user_factors_list, columns=['userId'] + [f'uf_svd_{i}' for i in range(self.N_FACTORS_SVD)]
            ).set_index('userId')
    
            # Extract item factors
            item_factors_list = []
            for inner_iid in full_trainset_svd.all_items():
                raw_iid = full_trainset_svd.to_raw_iid(inner_iid)
                factors = svd_model.qi[inner_iid]
                item_factors_list.append([raw_iid] + factors.tolist())
            item_factors_df = pd.DataFrame(
                item_factors_list, columns=['movieId'] + [f'if_svd_{i}' for i in range(self.N_FACTORS_SVD)]
            ).set_index('movieId')  
    
            # User features 
            logger.info("Engineering user features.")
            user_stats = ratings_df.groupby('userId')['rating'].agg(['mean', 'count']).rename(
                columns={'mean': 'user_avg_rating', 'count': 'user_num_ratings'}
            )
            user_movie_years = pd.merge(ratings_df, movies_df[['movieId', 'year']], on='movieId')
            user_newness_pref_series = user_movie_years.groupby('userId')['year'].mean().rename('user_newness_pref')
    
            # User genre diversity via entropy
            user_genre_ratings = pd.merge(
                ratings_df[['userId', 'movieId']],
                movies_df[['movieId'] + self.one_hot_genre_columns],
                on='movieId'
            )
            user_genre_counts = user_genre_ratings.groupby('userId')[self.one_hot_genre_columns].sum()
    
            def calculate_genre_entropy(row):
                genre_counts_for_user = row[row > 0]
                if genre_counts_for_user.empty:
                    return 0
                probabilities = genre_counts_for_user / genre_counts_for_user.sum()
                return entropy(probabilities, base=2)
    
            user_genre_diversity_series = user_genre_counts.apply(calculate_genre_entropy, axis=1).rename('user_genre_diversity')
    
            user_features_df = (
                self.user_factors_df.merge(user_stats, on='userId', how='left')
                                    .merge(user_newness_pref_series, on='userId', how='left')
                                    .merge(user_genre_diversity_series, on='userId', how='left')
            ).fillna({
                'user_avg_rating': full_trainset_svd.global_mean,
                'user_num_ratings': 0,
                'user_newness_pref': self.median_year,
                'user_genre_diversity': 0
            }) 
    
            # Movie features frame
            logger.info("Finalizing movie features.")
            movie_meta_features_to_select = (
                ['movieId', 'num_genres', 'year', 'is_old_movie', 'is_recent_movie', 'movie_genre_avg_popularity']
                + one_hot_decade_columns + self.one_hot_genre_columns
            )
            base_movie_meta_features_df = movies_df[movie_meta_features_to_select].set_index('movieId')
    
            movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).rename(
                columns={'mean': 'movie_avg_rating', 'count': 'movie_num_ratings'}
            )
            self.movie_features_df = (
                item_factors_df.merge(movie_stats, on='movieId', how='left')
                               .merge(base_movie_meta_features_df, on='movieId', how='left')
            )
    
            # Fill NaNs 
            self.movie_features_df['movie_avg_rating'] = self.movie_features_df['movie_avg_rating'].fillna(full_trainset_svd.global_mean)
            self.movie_features_df['movie_num_ratings'] = self.movie_features_df['movie_num_ratings'].fillna(0).astype(int)
            self.movie_features_df['year'] = self.movie_features_df['year'].fillna(self.median_year).astype(float)
            self.movie_features_df['num_genres'] = self.movie_features_df['num_genres'].fillna(0).astype(int)
            self.movie_features_df['is_old_movie'] = self.movie_features_df['is_old_movie'].fillna(0).astype(int)
            self.movie_features_df['is_recent_movie'] = self.movie_features_df['is_recent_movie'].fillna(0).astype(int)
            self.movie_features_df['movie_genre_avg_popularity'] = self.movie_features_df['movie_genre_avg_popularity'].fillna(self.global_avg_movie_rating)
            for col in one_hot_decade_columns + self.one_hot_genre_columns:
                self.movie_features_df[col] = self.movie_features_df[col].fillna(0)
    
            # Ensure index is movieId for downstream lookups
            if 'movieId' in self.movie_features_df.columns:
                self.movie_features_df = self.movie_features_df.set_index('movieId')
    
            # XGBoost training data 
            logger.info("Training XGBoost model.")
            xgb_train_df = ratings_df_orig.merge(user_features_df, on='userId', how='left')
            xgb_train_df = xgb_train_df.merge(self.movie_features_df, on='movieId', how='left')
    
            y = xgb_train_df['rating']
            drop_cols = ['userId', 'movieId', 'rating', 'timestamp']
            if 'title' in xgb_train_df.columns:   drop_cols.append('title')
            if 'genres' in xgb_train_df.columns:  drop_cols.append('genres')
    
            X = xgb_train_df.drop(columns=drop_cols, errors='ignore').fillna(0)
            X.columns = [str(c) for c in X.columns]
            self.original_X_columns = X.columns.tolist()
    
            X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)
    
            # Keep references for explainers
            self.X_train_ref = X_train.copy()
            self.shap_background = X_train.sample(n=min(1000, len(X_train)), random_state=42)  # kmeans pool  
    
            # LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_train_ref.values,
                feature_names=self.X_train_ref.columns.tolist(),
                mode='regression',
                discretize_continuous=True,
                sample_around_instance=True,
                random_state=42
            )
    
            # Train XGBoost regressor 
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100, learning_rate=0.1, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
            self.model.fit(X_train, y_train)
    
            # SHAP explainer 
            try:
                bg = self.X_train_ref.sample(n=min(200, len(self.X_train_ref)), random_state=42).to_numpy()
                self.shap_explainer = shap.TreeExplainer(
                    self.model,
                    data=bg,
                    feature_perturbation="interventional",
                    model_output="raw"
                )
            except Exception:
                masker = shap.maskers.Independent(self.X_train_ref.to_numpy())
                self.shap_explainer = shap.Explainer(self.model, masker)
    
            # Quick test metric
            y_pred_test = self.model.predict(X_test)
            rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            logger.info(f"XGBoost Test RMSE: {rmse_test:.4f}")
    
            # Mappings & dtype hygiene 
            self.ratings_df = ratings_df
            self.movie_id_to_title = movies_df.set_index('movieId')['title'].astype(str).to_dict()
            self.title_to_movie_id = {title: mid for mid, title in self.movie_id_to_title.items()}
    
            # Align user factor index dtype with ratings_df userId (prevents empty intersections)
            try:
                self.user_factors_df.index = self.user_factors_df.index.astype(ratings_df['userId'].dtype)
            except Exception:
                pass
    
            self.is_trained = True
            logger.info("Model training completed successfully")
    
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise e
    
    def get_recommendations(self, user_ratings_input, n=5):
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
    
            # ---- Build user genre profile ----
            liked_movie_ids = [rec['movieId'] for rec in top_recommendations]
            user_genre_vec = np.zeros((1, len(self.one_hot_genre_columns)))
    
            if liked_movie_ids:
                valid_liked_ids = self.movie_features_df.index.intersection(liked_movie_ids)
                if not valid_liked_ids.empty:
                    user_genre_vec = self.movie_features_df.loc[
                        valid_liked_ids, self.one_hot_genre_columns
                    ].mean().values.reshape(1, -1)
    
            # ---- Select "middle-of-the-road" candidates ----
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
    
                # Use movie_avg_rating and movie_num_ratings
                global_avg = mf.get("movie_avg_rating", 3.5)
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
    
            # ---- Sort by "middle-of-the-roadness" ----
            if candidates:
                candidates.sort(key=lambda x: abs(x["predicted_rating"] - 3.5))
                baseline = random.choice(candidates[:10]) if len(candidates) > 10 else candidates[0]
            else:
                # Fallback: pick any non-top recommendation
                fallback_candidates = [rec for rec in all_recommendations if rec['movieId'] not in top_ids]
                if not fallback_candidates:
                    return None
                baseline = random.choice(fallback_candidates)
    
            # ---- Make it LOOK like a strong recommendation ----
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

    # ------------------------- LIME calibration helpers -------------------------
    def _candidate_kernel_widths(self):
        """Propose a small grid of kernel_width values based on feature spread."""
        if self.X_train_ref is None or self.X_train_ref.shape[0] == 0:
            return [0.25, 0.5, 1.0, 2.0, 4.0]
        stds = np.std(self.X_train_ref.values, axis=0)
        sigma_ref = float(np.median(stds)) if np.isfinite(stds).any() else 1.0
        return [0.25 * sigma_ref, 0.5 * sigma_ref, 1.0 * sigma_ref, 2.0 * sigma_ref, 4.0 * sigma_ref]

    def _lime_local_fidelity(self, explanation, predict_fn):
        """Compute local R^2 fidelity between LIME surrogate and black-box predictions."""
        Z = getattr(explanation, 'domain', None)
        if Z is None or len(Z) == 0:
            return float('-inf')
        g = getattr(explanation, 'local_pred', None)
        if g is None:
            return float('-inf')
        f = predict_fn(Z)
        f = np.array(f, dtype=float)
        g = np.array(g, dtype=float)
        ss_res = float(np.sum((f - g) ** 2))
        ss_tot = float(np.sum((f - np.mean(f)) ** 2)) if len(f) > 1 else 0.0
        if ss_tot == 0.0:
            return float('-inf')
        return 1.0 - ss_res / ss_tot

    def _explain_with_lime_tuned(self, fv_row, num_features=5, seeds=(7, 11, 19, 23, 31)):
        """Tune kernel_width by local fidelity and report stability of top-k features."""
    
        instance = fv_row.values
    
        def predict_fn(X):
            df = pd.DataFrame(X, columns=self.original_X_columns)
            return self.model.predict(df)
    
        # Safety: fallbacks if training ref is missing
        train_X = self.X_train_ref
        if train_X is None or len(getattr(train_X, "columns", [])) == 0:
            train_X = pd.DataFrame(np.zeros((1, len(self.original_X_columns))), columns=self.original_X_columns)
    
        # tune kernel width 
        best_exp, best_kw, best_r2 = None, None, float('-inf')
        for kw in self._candidate_kernel_widths():
            tuner_explainer = llt.LimeTabularExplainer(
                training_data=train_X.values,
                feature_names=train_X.columns.tolist(),
                mode='regression',
                discretize_continuous=False,
                sample_around_instance=True,
                kernel_width=kw,
                random_state=42
            )
            exp = tuner_explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features,
                model_regressor=Ridge(alpha=10.0)
            )
            r2 = self._lime_local_fidelity(exp, predict_fn)
            if r2 > best_r2:
                best_exp, best_kw, best_r2 = exp, kw, r2
    
        # Extract best features (if any)
        best_feats = best_exp.as_list() if best_exp is not None else []
        features = [{"feature": name, "contribution": float(w)} for name, w in best_feats]
        topk = [name for name, _ in best_feats]
    
        # -------- stability across seeds (rebuild explainer with best_kw each time) --------
        topk_sets = []
        for s in seeds:
            stab_explainer = llt.LimeTabularExplainer(
                training_data=train_X.values,
                feature_names=train_X.columns.tolist(),
                mode='regression',
                discretize_continuous=False,
                sample_around_instance=True,
                kernel_width=best_kw if best_kw is not None else 1.0,
                random_state=s
            )
            exp_s = stab_explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=num_features
            )
            topk_sets.append({name for name, _ in exp_s.as_list()})
    
        def jaccard(a, b):
            inter = len(a & b)
            union = len(a | b) or 1
            return inter / union
    
        base = set(topk)
        stability = float(np.mean([jaccard(base, s) for s in topk_sets])) if topk_sets else 0.0
    
        return {
            'type': 'lime',
            'features': features,
            'kernel_width': float(best_kw) if best_kw is not None else None,
            'local_R2': float(best_r2) if np.isfinite(best_r2) else None,
            'stability_jaccard': stability
        }

    def generate_shap_explanation(self, recommendation, user_ratings_input):
        """Generate SHAP explanation with additivity and ablation diagnostics (DenseData-safe)."""
        try:
            movieId = recommendation['movieId']
    
            # ---------- Build feature vector (unchanged logic) ----------
            liked_movie_ids = set()
            valid_ratings = []
            for title, rating in user_ratings_input:
                mid = self.title_to_movie_id.get(title)
                if mid:
                    liked_movie_ids.add(mid)
                    valid_ratings.append(rating)
    
            user_avg = np.mean(valid_ratings) if valid_ratings else 0.0
            user_count = len(valid_ratings)
    
            # Estimate SVD user factors from similar users (fallback to mean)
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
    
            if movieId not in self.movie_features_df.index:
                return None
    
            mf = self.movie_features_df.loc[movieId]
            row = {col: user_feats.get(col, mf.get(col, 0)) for col in self.original_X_columns}
            fv = pd.DataFrame([row], columns=self.original_X_columns).fillna(0)
    
            # ---------- Ensure a non-legacy SHAP explainer ----------
            if self.shap_explainer is None:
                try:
                    # Use a plain NumPy background to avoid shap.utils._legacy.DenseData
                    bg = self.X_train_ref.sample(
                        n=min(200, len(self.X_train_ref)), random_state=42
                    ).to_numpy()
                    self.shap_explainer = shap.TreeExplainer(
                        self.model,
                        data=bg,
                        feature_perturbation="interventional",
                        model_output="raw"
                    )
                except Exception:
                    # Fallback to the modern API with an explicit masker
                    masker = shap.maskers.Independent(self.X_train_ref.to_numpy())
                    self.shap_explainer = shap.Explainer(self.model, masker)
    
            # ---------- Compute SHAP values; support both APIs ----------
            try:
                # TreeExplainer-style returns ndarray (sometimes list-wrapped)
                shap_values = self.shap_explainer.shap_values(fv, check_additivity=True)
                if isinstance(shap_values, list):
                    phi = np.array(shap_values[0][0], dtype=float)
                else:
                    phi = np.array(shap_values[0], dtype=float)
    
                base = self.shap_explainer.expected_value
                if isinstance(base, (list, np.ndarray)):
                    base = float(base[0])
                else:
                    base = float(base)
            except TypeError:
                # New Explainer API returns an Explanation object
                explanation = self.shap_explainer(fv)
                phi = np.array(explanation.values[0], dtype=float)
                base = explanation.base_values[0] if hasattr(explanation, "base_values") \
                       else float(np.mean(self.model.predict(self.X_train_ref)))
    
            # ---------- Additivity check ----------
            pred = float(self.model.predict(fv)[0])
            additivity_resid = abs((base + phi.sum()) - pred)
    
            # ---------- Quick ablation faithfulness ----------
            fv_drop = fv.copy()
            top_idx = np.argsort(-np.abs(phi))[:3]
            bg_means = pd.Series(self.X_train_ref.mean(), index=self.original_X_columns) \
                       if self.X_train_ref is not None else pd.Series(0, index=self.original_X_columns)
            for j in top_idx:
                col = self.original_X_columns[j]
                fv_drop.iloc[0, j] = float(bg_means[col])
    
            pred_drop = float(self.model.predict(fv_drop)[0])
            delta_pred = pred - pred_drop
            delta_attr = float(np.sum(phi[top_idx]))
    
            # ---------- Assemble top features ----------
            feature_contributions = list(zip(fv.columns, phi))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            significant_features = [
                {
                    'feature': feat,
                    'contribution': float(contrib),
                    'feature_value': float(fv[feat].iloc[0])
                }
                for feat, contrib in feature_contributions[:5]
                if abs(contrib) > 1e-6
            ]
    
            return {
                'type': 'shap',
                'baseline': base,
                'prediction': pred,
                'additivity_residual': float(additivity_resid),
                'ablation_delta_pred': float(delta_pred),
                'ablation_delta_attr_sum': float(delta_attr),
                'features': significant_features
            }
    
        except Exception as e:
            logger.warning(f"Could not generate SHAP explanation: {str(e)}")
            return None
    
    def generate_lime_explanation(self, recommendation, user_ratings_input):
        """Generate LIME explanation with tuned kernel_width and fidelity/stability metrics."""
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
                row = {col: user_feats.get(col, mf.get(col, 0)) for col in self.original_X_columns}
                fv = pd.DataFrame([row], columns=self.original_X_columns).fillna(0)

                # Use tuned LIME with fidelity and stability metrics
                return self._explain_with_lime_tuned(fv.iloc[0], num_features=5)
        
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
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {"role": "system", "content": "You are a helpful movie recommendation explainer. Keep explanations brief and personalized."},
                    {"role": "user", "content": prompt}
                ],
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
        """Get recommendations with SHAP + LIME explanations."""
        # Base recommendations
        recommendations = self.get_recommendations(user_ratings_input, n)
        if not include_explanations or not recommendations:
            return recommendations
    
        # Build lightweight user features once per request
        liked_movie_ids, valid_ratings = set(), []
        for title, rating in user_ratings_input:
            mid = self.title_to_movie_id.get(title) if hasattr(self, "title_to_movie_id") else None
            if mid:
                liked_movie_ids.add(mid)
                valid_ratings.append(rating)

        user_avg = float(np.mean(valid_ratings)) if valid_ratings else 0.0
        user_count = len(valid_ratings)
        sim_users = self.find_similar_users(liked_movie_ids, self.ratings_df, 4.0) if len(liked_movie_ids) else []

        if len(sim_users) < 5 or self.user_factors_df is None or self.user_factors_df.empty:
            est_factors = self.user_factors_df.mean(numeric_only=True) if self.user_factors_df is not None else pd.Series(dtype=float)
        else:
            valid_sim_users = self.user_factors_df.index.intersection(sim_users)
            if getattr(valid_sim_users, "empty", False):
                est_factors = self.user_factors_df.mean(numeric_only=True)
            else:
                sim_df = self.user_factors_df.loc[valid_sim_users]
                est_factors = sim_df.mean(numeric_only=True)

        # Assemble common user features dict
        user_feats = {
            'user_avg_rating': user_avg,
            'user_num_ratings': user_count,
            'user_newness_pref': getattr(self, "median_year", 2000),
            'user_genre_diversity': 0.0,
        }
        # Ensure uf_svd_* present using the actual index names
        if isinstance(est_factors, pd.Series):
            for k, v in est_factors.items():
                user_feats[str(k)] = float(v)

        for rec in recommendations:
            movieId = rec['movieId']
            if movieId not in self.movie_features_df.index:
                rec['explanations'] = None
                continue

            # Build feature vector aligned to training columns
            mf = self.movie_features_df.loc[movieId]
            row = {col: user_feats.get(col, float(mf.get(col, 0))) for col in self.original_X_columns}
            fv = pd.DataFrame([row], columns=self.original_X_columns).fillna(0)

            explanations_payload = {}

            # ---------- SHAP ----------
            try:
                explainer = getattr(self, "shap_explainer", None)
                if explainer is None:
                    try:
                        bg_src = self.shap_background if self.shap_background is not None else self.X_train_ref
                        bg = shap.kmeans(bg_src.to_numpy() if hasattr(bg_src, "to_numpy") else np.asarray(bg_src), 100)
                    except Exception:
                        bg_src = self.shap_background if self.shap_background is not None else self.X_train_ref
                        bg_df = bg_src.sample(n=min(200, len(bg_src))) if hasattr(bg_src, "sample") else bg_src
                        bg = bg_df.to_numpy() if hasattr(bg_df, "to_numpy") else np.asarray(bg_df)

                    self.shap_explainer = shap.TreeExplainer(
                        self.model,
                        data=bg,
                        feature_perturbation="interventional",
                        model_output="raw"
                    )
                    explainer = self.shap_explainer

                shap_values = explainer.shap_values(fv, check_additivity=True)
                if isinstance(shap_values, list):
                    phi = np.array(shap_values[0][0], dtype=float)
                else:
                    phi = np.array(shap_values[0], dtype=float)

                expected = explainer.expected_value
                base = float(expected if not isinstance(expected, (list, np.ndarray)) else expected[0])
                pred = float(self.model.predict(fv)[0])
                add_resid = float(abs((base + float(phi.sum())) - pred))

                feature_contributions = sorted(zip(fv.columns, phi), key=lambda x: abs(x[1]), reverse=True)
                shap_top = [
                    {'feature': name, 'contribution': float(w), 'feature_value': float(fv.at[0, name])}
                    for name, w in feature_contributions[:5] if abs(w) > 1e-6
                ]
                explanations_payload['shap'] = {
                    'baseline': base,
                    'pred': pred,
                    'additivity_residual': add_resid,
                    'top_features': shap_top
                }
            except Exception as e:
                logger.warning(f"Could not generate SHAP explanation for {movieId}: {e}")
                explanations_payload['shap'] = None

            # ---------- LIME ----------
            try:

                lime_exp = self._explain_with_lime_tuned(fv.iloc[0], num_features=5)
                lime_feats = (lime_exp.get('features') or lime_exp.get('top_features') or [])
            
                # Convert to SHAP-like items: [{'feature', 'contribution', 'feature_value'}]
                lime_top = []
                for f in lime_feats[:5]:
                    name = str(f.get('feature'))
                    contrib = float(f.get('contribution', 0.0))
                    # Try to provide feature_value like SHAP does (UI doesn’t need it but keep parity)
                    base_col = re.split(r'<=|>=|>|<', name)[0].strip()
                    fv_val = float(fv[base_col].iloc[0]) if base_col in fv.columns else None
                    lime_top.append({
                        'feature': name,
                        'contribution': contrib,
                        'feature_value': fv_val
                    })
            
                # Mirror SHAP writeback: explanation (singular) + explanation_type
                rec['explanation'] = {
                    'type': 'lime',
                    'features': lime_top,
                    # Keep keys aligned with SHAP payload; use None where not applicable
                    'baseline': None,
                    'pred': float(self.model.predict(fv)[0]),
                    'additivity_residual': None
                }
                rec['explanation_type'] = 'lime'
            
            except Exception as e:
                logger.warning(f"Could not generate LIME explanation for {movieId}: {e}")
                # Still emit a well-formed, SHAP-shaped object so the UI renders the card
                rec['explanation'] = {
                    'type': 'lime',
                    'features': [],
                    'baseline': None,
                    'pred': float(self.model.predict(fv)[0]),
                    'additivity_residual': None
                }
                rec['explanation_type'] = 'lime'

        return recommendations

