# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:37:06 2025

@author: freez
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from collections import defaultdict
import shap
import lime
import lime.lime_tabular
import xgboost as xgb
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Initial Data Loading 
movies_df_orig = pd.read_csv('dataset/100kDataset/movies.csv')
ratings_df_orig = pd.read_csv('dataset/100kDataset/ratings.csv')

# Create copies to work with to preserve originals if needed for direct inspection
movies_df = movies_df_orig.copy()
ratings_df = ratings_df_orig.copy()

# Remove movies with less than 3 ratings to remove noise
movie_rating_counts = ratings_df.groupby('movieId')['rating'] \
                                .count() \
                                .rename('movie_num_ratings') \
                                .reset_index()   
                                
cutoff_list = movie_rating_counts.loc[
    movie_rating_counts['movie_num_ratings'] < 3,
    'movieId'
].tolist()         
   
movies_df = movies_df[~movies_df['movieId'].isin(cutoff_list)].reset_index(drop=True) 
ratings_df = ratings_df[~ratings_df['movieId'].isin(cutoff_list)].reset_index(drop=True)           
                                
print("\nOriginal Movies head:")
print(movies_df.head())
print("\nOriginal Ratings head:")
print(ratings_df.head())

# Getting average rating per genre
print("\nCalculating average rating per genre")
movie_ratings_for_genre_avg = pd.merge(movies_df, ratings_df, on='movieId')
temp_genre_df = movie_ratings_for_genre_avg[['movieId', 'genres', 'rating']].copy()
temp_genre_df['genres_list'] = temp_genre_df['genres'].str.split('|')
exploded_genres_ratings = temp_genre_df.explode('genres_list')
genre_avg_ratings_series = exploded_genres_ratings.groupby('genres_list')['rating'].mean()
genre_to_avg_rating_map = genre_avg_ratings_series.to_dict()
print(f"{len(genre_to_avg_rating_map)} genres identified")
# print("Example genre average ratings:", dict(list(genre_to_avg_rating_map.items())[:5]))
# Global average movie rating for fallback
global_avg_movie_rating = ratings_df['rating'].mean()


# Movie-Level Feature Engineering
print("\nMovie-Level Feature Engineering on movies_df")

# Add num_genres
movies_df['num_genres'] = movies_df['genres'].apply(lambda x: len(x.split('|')) if x != '(no genres listed)' else 0)

# Add year (from title)
movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
median_year = movies_df['year'].median() # Calculate median before NaNs from non-extractable titles
movies_df['year'] = movies_df['year'].fillna(median_year) # Fill NaNs for year

movies_df['is_old_movie'] = (movies_df['year'] < 1990).astype(int)
movies_df['is_recent_movie'] = (movies_df['year'] > 2015).astype(int)

# Add and encode decade
movies_df['decade'] = (movies_df['year'] // 10) * 10
movies_df['decade'] = movies_df['decade'].astype(int) # Ensure decade is int for get_dummies
decade_dummies = pd.get_dummies(movies_df['decade'], prefix='decade', dummy_na=False, dtype=int) # dummy_na=False to not create decade_nan
movies_df = pd.concat([movies_df, decade_dummies], axis=1)
one_hot_decade_columns = decade_dummies.columns.tolist()

# Compute Series indexed by movieId
avg_series = ratings_df.groupby('movieId')['rating'].mean()
count_series = ratings_df.groupby('movieId')['rating'].count()

# Map onto movies_df
movies_df['avg_rating']  = movies_df['movieId'].map(avg_series).fillna(global_avg_movie_rating)
movies_df['num_ratings'] = movies_df['movieId'].map(count_series).fillna(0).astype(int)


# Add movie_genre_avg_popularity
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
    lambda x: calculate_movie_genre_avg_pop(x, genre_to_avg_rating_map, global_avg_movie_rating)
)

# One-hot encode genres (is_<genre>)
genres_dummies_movies = movies_df['genres'].str.get_dummies(sep='|')

# Ensure no clashes with other column names, e.g. if a genre is named 'year'
genres_dummies_movies.columns = [f"genre_{col.replace(' ', '_').replace('-', '_')}" for col in genres_dummies_movies.columns]
one_hot_genre_columns = genres_dummies_movies.columns.tolist() # Update this list
movies_df = pd.concat([movies_df, genres_dummies_movies], axis=1)

# Columns to keep from movies_df for merging into the main feature set later
movie_meta_features_to_select = ['movieId', 'num_genres', 'year', 'is_old_movie', 'is_recent_movie', 'movie_genre_avg_popularity'] + \
                                one_hot_decade_columns + one_hot_genre_columns

# Main DataFrame for SVD and initial user/item stats
# This df exists for Surprise and initial user/item stats
df = pd.merge(ratings_df, movies_df[['movieId', 'title']], on='movieId')
print("\nMain df for Surprise (first few rows):")
print(df.head()) # df now is ratings + movie titles

# Get user, item, rating for SVD
reader = Reader(rating_scale=(0.5, 5.0))
surprise_data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
print("\nData loaded into Surprise Dataset format.")


# SVD Feature Generation
print("\nSVD Feature Generation & Initial Setup")
N_FACTORS_SVD = 10
svd_model = SVD(n_factors=N_FACTORS_SVD, n_epochs=20, random_state=42, verbose=False)
full_trainset_svd = surprise_data.build_full_trainset()
print("Training SVD model on the full dataset to extract latent factors...")
svd_model.fit(full_trainset_svd)
print("SVD model training complete.")

user_factors_list = []
for inner_uid in full_trainset_svd.all_users():
    raw_uid = full_trainset_svd.to_raw_uid(inner_uid)
    factors = svd_model.pu[inner_uid]
    user_factors_list.append([raw_uid] + factors.tolist())
user_factors_df = pd.DataFrame(user_factors_list, columns=['userId'] + [f'uf_svd_{i}' for i in range(N_FACTORS_SVD)])
user_factors_df = user_factors_df.set_index('userId')

item_factors_list = []
for inner_iid in full_trainset_svd.all_items():
    raw_iid = full_trainset_svd.to_raw_iid(inner_iid)
    factors = svd_model.qi[inner_iid]
    item_factors_list.append([raw_iid] + factors.tolist())
item_factors_df = pd.DataFrame(item_factors_list, columns=['movieId'] + [f'if_svd_{i}' for i in range(N_FACTORS_SVD)])
item_factors_df = item_factors_df.set_index('movieId')

print(f"Extracted {len(user_factors_df)} user SVD factors and {len(item_factors_df)} item SVD factors.")

# Mappings 
movie_id_to_title = movies_df_orig.set_index('movieId')['title'].to_dict() 
title_to_movie_id = {title: id for id, title in movie_id_to_title.items()}


# User-Level Feature Engineering 
print("\nUser-Level Feature Engineering")
# Basic user stats (avg rating, num ratings) calculated from original ratings_df
user_stats = ratings_df.groupby('userId')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'user_avg_rating', 'count': 'user_num_ratings'})

# user_newness_pref: Average year of movies rated by user
user_movie_years = pd.merge(ratings_df, movies_df[['movieId', 'year']], on='movieId')
user_newness_pref_series = user_movie_years.groupby('userId')['year'].mean().rename('user_newness_pref')

# user_genre_diversity: Entropy of user's rated genres
# Merge user ratings with movie genres (from movies_df)
user_genre_ratings = pd.merge(ratings_df[['userId', 'movieId']], movies_df[['movieId'] + one_hot_genre_columns], on='movieId')
user_genre_counts = user_genre_ratings.groupby('userId')[one_hot_genre_columns].sum() # Sum of 1s gives count of ratings per genre for user


def calculate_genre_entropy(row):
    genre_counts_for_user = row[row > 0] # Filter out genres not rated by the user
    if genre_counts_for_user.empty:
        return 0
    probabilities = genre_counts_for_user / genre_counts_for_user.sum()
    return entropy(probabilities, base=2)

user_genre_diversity_series = user_genre_counts.apply(calculate_genre_entropy, axis=1).rename('user_genre_diversity')

# Merge user features: SVD factors, basic stats, newness pref, genre diversity
user_features_df = user_factors_df.merge(user_stats, on='userId', how='left')
user_features_df = user_features_df.merge(user_newness_pref_series, on='userId', how='left')
user_features_df = user_features_df.merge(user_genre_diversity_series, on='userId', how='left')

# Fill NaNs for user features
user_features_df = user_features_df.fillna({
    'user_avg_rating': full_trainset_svd.global_mean, 
    'user_num_ratings': 0,
    'user_newness_pref': median_year, # Fallback to median movie year
    'user_genre_diversity': 0 # Fallback for users with no ratings or diverse genres
})
print("User features (user_features_df) head:")
print(user_features_df.head())


# Movie-Level Feature Engineering 
print("\nConstructing full movie_features_df")
movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'movie_avg_rating', 'count': 'movie_num_ratings'})

# Base movie features from the enhanced movies_df
# Selecting defined columns: movie_meta_features_to_select
base_movie_meta_features_df = movies_df[movie_meta_features_to_select].set_index('movieId')

# Merge SVD item factors with movie_stats
movie_features_df = item_factors_df.merge(movie_stats, on='movieId', how='left')
# Merge with base movie meta features (num_genres, year, is_old, is_recent, decades, genre_avg_pop, one-hot genres)
movie_features_df = movie_features_df.merge(base_movie_meta_features_df, on='movieId', how='left')

# Fill NaNs for movie features
movie_features_df['movie_avg_rating'] = movie_features_df['movie_avg_rating'].fillna(full_trainset_svd.global_mean)
movie_features_df['movie_num_ratings'] = movie_features_df['movie_num_ratings'].fillna(0)
# Fill NaNs for new movie features that might not have matched (ex a movieId is in SVD factors but not movies_df)
movie_features_df['year'] = movie_features_df['year'].fillna(median_year) # Already filled in movies_df, but as safety
movie_features_df['num_genres'] = movie_features_df['num_genres'].fillna(0)
movie_features_df['is_old_movie'] = movie_features_df['is_old_movie'].fillna(False)
movie_features_df['is_recent_movie'] = movie_features_df['is_recent_movie'].fillna(False)
movie_features_df['movie_genre_avg_popularity'] = movie_features_df['movie_genre_avg_popularity'].fillna(global_avg_movie_rating)
for col in one_hot_decade_columns: # Fill NaN for decade columns with 0
    movie_features_df[col] = movie_features_df[col].fillna(0)
for col in one_hot_genre_columns: # Fill NaN for one-hot genre columns with 0
    movie_features_df[col] = movie_features_df[col].fillna(0)

# Construct XGBoost Training Set
print("\nConstructing XGBoost Training Set with new features")
# Merge ratings_df with user_features_df and movie_features_df
xgb_train_df = ratings_df_orig.merge(user_features_df, on='userId', how='left')
xgb_train_df = xgb_train_df.merge(movie_features_df, on='movieId', how='left')

# Define target variable y
y = xgb_train_df['rating']

# Define features X
# Columns to drop: identifiers, raw genre string, title, timestamp
columns_to_drop_for_X = ['userId', 'movieId', 'rating', 'timestamp'] 
# Add any other non-feature columns that might have been included, e.g. 'genres' string if it's there
if 'title' in xgb_train_df.columns: columns_to_drop_for_X.append('title')
if 'genres' in xgb_train_df.columns: columns_to_drop_for_X.append('genres')

X = xgb_train_df.drop(columns=columns_to_drop_for_X, errors='ignore')

# Handle any remaining NaNs in X before training
X = X.fillna(0)
print(f"\nNumber of NaNs added: {X.isnull().sum().sum()}")

# Ensure all feature names are strings
X.columns = [str(col) for col in X.columns]

# Split data
X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor 
print("\nXGBoost train start")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
print("XGBoost train finish")
y_pred_test = xgb_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"XGBoost Test RMSE: {rmse_test:.4f}")


# Recommendation Generation & Explainability
print("\nRecommendation and Explanation functions will use the new feature set.")

MIN_RATINGS_THRESHOLD = 10 
ABS_PRED_RATING_THRESHOLD = 4.0

def get_new_user_recommendations(
    new_user_ratings_input,
    n,
    model,
    all_movie_features_df,
    historical_user_factors_df,
    original_X_columns,
    title_to_movie_id_map,
    movie_id_to_title_map,
    ratings_df,
    genre_columns,
    min_ratings_threshold=10,
    abs_pred_rating_threshold=4.0,
    min_similar_rating=4.0,
    min_similar_users=5,
    genre_similarity_threshold=0.1
):
    """
    Hybrid recs for a cold-start: 
      1) Estimate user SVD via similar users
      2) Compute genre‐profile & similarity
      3) Rank by uplift × genre_similarity
    """
    liked_movie_ids = set()
    valid_ratings = []
    for title, rating in new_user_ratings_input:
        mid = title_to_movie_id_map.get(title)
        if mid:
            liked_movie_ids.add(mid)
            valid_ratings.append(rating)
        else:
            print(f"Warning: '{title}' not found, skipping.")
    if not liked_movie_ids:
        return []

    user_avg = np.mean(valid_ratings) if valid_ratings else 0.0
    user_count = len(valid_ratings)

    # Find similar users by overlap
    def find_similar_users(lm_ids, ratings_df, thresh=4.0):
        users = set()
        for m_id in lm_ids:
            dfm = ratings_df[(ratings_df.movieId == m_id) & (ratings_df.rating >= thresh)]
            users.update(dfm.userId.tolist())
        return list(users)

    sim_users = find_similar_users(liked_movie_ids, ratings_df, min_similar_rating)
    if len(sim_users) < min_similar_users:
        est_factors = historical_user_factors_df.mean() # Fallback to mean SVD factors
    else:
        # Filter historical_user_factors_df for users present in its index
        valid_sim_users = historical_user_factors_df.index.intersection(sim_users)
        if not valid_sim_users.empty:
            sim_df = historical_user_factors_df.loc[valid_sim_users]
            est_factors = sim_df.mean()
        else: # Fallback if no similar users are found in historical_user_factors_df (edge case)
             est_factors = historical_user_factors_df.mean()

    # Build genre profile
    def make_genre_profile(lm_ids, mf_df, gcols):
        # Filter mf_df for movies present in its index
        valid_lm_ids = mf_df.index.intersection(lm_ids)
        if not valid_lm_ids.empty:
            sub = mf_df.loc[valid_lm_ids, gcols]
            return sub.mean().values.reshape(1, -1)
        else: # Fallback if no liked movies are found in mf_df (edge case)
            return np.zeros((1, len(gcols)))

    user_genre_vec = make_genre_profile(liked_movie_ids, all_movie_features_df, genre_columns)

    # Assemble new‐user feature dict
    user_feats = {
        'user_avg_rating': user_avg,
        'user_num_ratings': user_count,
        'user_newness_pref': median_year, # Default, can be refined
        'user_genre_diversity': 0 # Default, can be refined
    }
    # Add estimated SVD factors to user_feats
    for i, factor_val in enumerate(est_factors): # est_factors is a Series
        user_feats[f'uf_svd_{i}'] = factor_val
    
    # Ensure all SVD factor columns (uf_svd_0 to uf_svd_N-1) expected by the model are in user_feats
    for i in range(N_FACTORS_SVD): # Assuming N_FACTORS_SVD is globally available or passed
        col_name = f'uf_svd_{i}'
        if col_name not in user_feats:
            user_feats[col_name] = 0 # Default to 0 if not estimated

    # Score candidates
    cands = []
    for mid in all_movie_features_df.index:
        if mid in liked_movie_ids:
            continue
        mf = all_movie_features_df.loc[mid]
        if mf.get('movie_num_ratings', 0) < min_ratings_threshold:
            continue

        # genre similarity
        mv = mf[genre_columns].values.reshape(1, -1)
        sim = float(cosine_similarity(user_genre_vec, mv)[0,0])
        if sim < genre_similarity_threshold:
            continue

        # build feature vector
        row = {
            col: user_feats.get(col, mf.get(col, 0))
            for col in original_X_columns
        }
        fv = pd.DataFrame([row], columns=original_X_columns).fillna(0)

        pr = model.predict(fv)[0]
        if pr < abs_pred_rating_threshold:
            continue

        avg = mf.get('movie_avg_rating', 3.5)
        uplift = pr - avg
        adj = uplift * sim

        cands.append({
            'movieId': mid,
            'predicted_rating': pr,
            'uplift': uplift,
            'genre_similarity': sim,
            'adjusted_uplift': adj,
            'feature_vector': fv
        })

    # Return top‐N by adjusted uplift
    cands.sort(key=lambda x: x['adjusted_uplift'], reverse=True)
    return [
        {
            'title': movie_id_to_title_map.get(x['movieId'], 'Unknown'),
            'predicted_rating': x['predicted_rating'],
            'uplift': x['uplift'],
            'genre_similarity': x['genre_similarity'],
            'adjusted_uplift': x['adjusted_uplift'],
            'movieId': x['movieId'],
            'feature_vector': x['feature_vector']
        }
        for x in cands[:n]
    ]


# SHAP and LIME functions (get_feature_vector_for_explanation, etc.) 
shap_explainer_xgb = None # Will be initialized in __main__ if run
lime_explainer_xgb = None # Will be initialized in __main__ if run

def get_feature_vector_for_explanation(user_id_str, movie_id_for_explanation, user_features, movie_features, original_X_cols):
    user_id = int(user_id_str)
    if user_id not in user_features.index or movie_id_for_explanation not in movie_features.index:
        print("User or Movie ID not found for explanation")
        return None
    user_f = user_features.loc[user_id]
    movie_f = movie_features.loc[movie_id_for_explanation]
    combined_data = {}
    for col in original_X_cols:
        if col in user_f.index: combined_data[col] = user_f[col]
        elif col in movie_f.index: combined_data[col] = movie_f[col]
        else: combined_data[col] = 0
    return pd.DataFrame([combined_data], columns=original_X_cols).fillna(0)

def explain_xgb_recommendation_with_shap(feature_vector_df_shap, shap_explainer_to_use):
    if feature_vector_df_shap is None or feature_vector_df_shap.empty or shap_explainer_to_use is None:
        return None, None
    shap_values_instance = shap_explainer_to_use.shap_values(feature_vector_df_shap)
    return shap_values_instance[0], shap_explainer_to_use.expected_value

def lime_predict_fn_xgb(data_for_lime, columns_for_lime): # Pass columns
    return xgb_model.predict(pd.DataFrame(data_for_lime, columns=columns_for_lime))

def explain_xgb_recommendation_with_lime(feature_vector_df_lime, lime_explainer_to_use, predict_fn_for_lime, num_lime_features=10, columns_for_lime=None):
    if feature_vector_df_lime is None or feature_vector_df_lime.empty or lime_explainer_to_use is None or columns_for_lime is None:
        return None
    instance_to_explain_lime = feature_vector_df_lime.iloc[0].values
    # Pass the columns to the predict_fn using a lambda or functools.partial if it doesn't accept extra args directly
    bound_predict_fn = lambda x: predict_fn_for_lime(x, columns_for_lime)
    explanation = lime_explainer_to_use.explain_instance(
        data_row=instance_to_explain_lime,
        predict_fn=bound_predict_fn, # Use the bound function
        num_features=num_lime_features
    )
    return explanation


if __name__ == '__main__':
    # Re-initialize explainers with the new X_train
    shap_explainer_xgb = shap.TreeExplainer(xgb_model, data=X_train) 
    lime_explainer_xgb = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values, 
        feature_names=X_train.columns.tolist(),
        class_names=['predicted_rating'],
        mode='regression',
        random_state=42
    )

    # Test new user recommendations 
    print("\n\nNew User Recommendations")
    new_user_input = [
        ("Freddy vs. Jason (2003)", 5.0),
        ("Shining, The (1980)", 5.0),
        ("Nightmare on Elm Street 2: Freddy's Revenge, A (1985)", 5.0), 
        ("Pet Sematary (1989)", 5.0), 
        ("Friday the 13th (2009)", 5.0)
    ]
    
    new_user_recs = get_new_user_recommendations(
        new_user_input, 5,
        xgb_model,
        movie_features_df,
        user_factors_df,
        X_train.columns,
        title_to_movie_id,
        movie_id_to_title,
        ratings_df,
        one_hot_genre_columns
    )
    
    if new_user_recs:
        print(f"\nTop 5 recommendations for the new user:")
        for rec in new_user_recs:
            print(f"- {rec['title']} (Pred Rating: {rec['predicted_rating']:.4f}, Uplift: {rec['uplift']:.4f}) (MovieID: {rec['movieId']})")
    else:
        print("Could not generate recommendations for the new user.")

    # 1) Collect all candidate feature-vectors
    #    (assumes each rec dict has a 'feature_vector' key with a 1×F DataFrame)
    candidate_fvs = [rec['feature_vector'] for rec in new_user_recs]
    all_candidates_df = pd.concat(candidate_fvs, axis=0)
    
    # 2) Build a SHAP background sample from this new-user space
    bg_size = min(len(all_candidates_df), 100)
    background_new_user = all_candidates_df.sample(n=bg_size, random_state=42).astype(np.float64)
    
    # 3) Instantiate per-user SHAP explainer
    shap_explainer_new = shap.TreeExplainer(xgb_model, data=background_new_user)
    
    # 4) Instantiate per-user LIME explainer
    lime_explainer_new = lime.lime_tabular.LimeTabularExplainer(
        training_data=all_candidates_df.values.astype(np.float64),
        feature_names=X_train.columns.tolist(),
        mode='regression',
        random_state=42
    )
    
    # 5) Explain each recommendation
    print("\n--- New-User SHAP & LIME Explanations ---")
    for rec in new_user_recs:
        title    = rec['title']
        fv        = rec['feature_vector']  # 1×F DataFrame
    
        # 5a) SHAP
        shap_vals = shap_explainer_new.shap_values(fv)
        base_val  = shap_explainer_new.expected_value
        contribs  = list(zip(fv.columns, shap_vals[0]))
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    
        print(f"\n{title}:")
        print(f"  SHAP baseline: {base_val:.3f}")
        print("  Top SHAP drivers:")
        for feat, val in contribs[:5]:
            sign = "+" if val >= 0 else ""
            print(f"    {sign}{val:.3f} → {feat}")
    
        # 5b) LIME
        lime_exp = lime_explainer_new.explain_instance(
            fv.values[0],
            lambda x: xgb_model.predict(pd.DataFrame(x, columns=fv.columns)),
            num_features=5
        )
        print("  LIME feature weights:")
        for feat, weight in lime_exp.as_list():
            sign = "+" if weight >= 0 else ""
            print(f"    {sign}{weight:.3f} → {feat}")
        
    
