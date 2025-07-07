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

# Initial Data Loading
try:
    movies_df_orig = pd.read_csv('../dataset/100kDataset/movies.csv')
    ratings_df_orig = pd.read_csv('../dataset/100kDataset/ratings.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("WARNING: Dataset files not found. Proceeding with empty DataFrames for testing purposes.")
    print("Please ensure '../dataset/100kDataset/movies.csv' and '../dataset/100kDataset/ratings.csv' are available for full functionality.")
    # Define expected columns for empty DataFrames to allow script execution
    movies_df_orig = pd.DataFrame(columns=['movieId', 'title', 'genres'])
    ratings_df_orig = pd.DataFrame(columns=['userId', 'movieId', 'rating', 'timestamp'])

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
N_FACTORS_SVD = 50
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
if X.empty or y.empty:
    print("WARNING: Training data (X or y) is empty. Skipping model training and recommendation generation.")
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    xgb_model = None # Ensure xgb_model is defined, even if None
    y_pred_test = None # Also define other variables that would be created in the else block
    rmse_test = None
else:
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
      4) Include SHAP and LIME explanations.
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

    def find_similar_users(lm_ids, ratings_df, thresh=4.0):
        users = set()
        for m_id in lm_ids: # Corrected variable name
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


    def make_genre_profile(lm_ids, mf_df, gcols):
        # Filter mf_df for movies present in its index
        valid_lm_ids = mf_df.index.intersection(lm_ids)
        if not valid_lm_ids.empty:
            sub = mf_df.loc[valid_lm_ids, gcols]
            return sub.mean().values.reshape(1, -1)
        else: # Fallback if no liked movies are found in mf_df (edge case)
            return np.zeros((1, len(gcols)))


    user_genre_vec = make_genre_profile(liked_movie_ids, all_movie_features_df, genre_columns)

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


    cands = []
    global shap_explainer_xgb, lime_explainer_xgb # Access global explainers

    for mid_candidate in all_movie_features_df.index:
        if mid_candidate in liked_movie_ids:
            continue

        movie_candidate_features = all_movie_features_df.loc[mid_candidate]
        if movie_candidate_features.get('movie_num_ratings', 0) < min_ratings_threshold:
            continue

        movie_genre_vec = movie_candidate_features[genre_columns].values.reshape(1, -1)
        genre_sim = float(cosine_similarity(user_genre_vec, movie_genre_vec)[0,0])
        if genre_sim < genre_similarity_threshold:
            continue

        # Construct feature vector for this user-movie pair
        feature_row_dict = {}
        for col in original_X_columns:
            if col in user_feats:
                feature_row_dict[col] = user_feats[col]
            elif col in movie_candidate_features.index: # movie_candidate_features is a Series
                feature_row_dict[col] = movie_candidate_features[col]
            else:
                feature_row_dict[col] = 0 # Default for any other columns (e.g. interaction features not applicable here)

        # Ensure DataFrame has all original_X_columns in the correct order
        feature_vector_df = pd.DataFrame([feature_row_dict], columns=original_X_columns).fillna(0)
        # Reorder columns to match X_train's column order explicitly
        feature_vector_df = feature_vector_df[original_X_columns]


        predicted_rating = model.predict(feature_vector_df)[0]

        if predicted_rating < abs_pred_rating_threshold:
            continue

        movie_avg_rating = movie_candidate_features.get('movie_avg_rating', global_avg_movie_rating) # Use global avg as fallback
        uplift_score = predicted_rating - movie_avg_rating
        adjusted_uplift_score = uplift_score * genre_sim

        shap_explanation_output = None
        shap_base = None
        lime_explanation_output = None

        if shap_explainer_xgb:
            shap_explanation_output, shap_base, _ = explain_xgb_recommendation_with_shap(
                feature_vector_df, shap_explainer_xgb, original_X_columns
            )

        if lime_explainer_xgb:
            lime_explanation_output = explain_xgb_recommendation_with_lime(
                feature_vector_df, lime_explainer_xgb, num_lime_features=10
            )

        cands.append({
            'movieId': mid_candidate,
            'title': movie_id_to_title_map.get(mid_candidate, 'Unknown Title'),
            'predicted_rating': predicted_rating,
            'uplift': uplift_score,
            'genre_similarity': genre_sim,
            'adjusted_uplift': adjusted_uplift_score,
            'shap_explanation': shap_explanation_output,
            'shap_base_value': shap_base,
            'lime_explanation': lime_explanation_output
        })

    cands.sort(key=lambda x: x['adjusted_uplift'], reverse=True)
    return cands[:n] # Return the full dictionary objects


# SHAP and LIME functions
shap_explainer_xgb = None
lime_explainer_xgb = None

# Renamed, as this is for existing users. New user explanation uses fv from get_new_user_recommendations
def get_feature_vector_for_existing_user_explanation(user_id_str, movie_id_for_explanation, user_features, movie_features, original_X_cols):
    user_id = int(user_id_str)
    if user_id not in user_features.index or movie_id_for_explanation not in movie_features.index:
        print(f"User {user_id} or Movie {movie_id_for_explanation} not found for explanation.")
        return None

    user_f_series = user_features.loc[user_id]
    movie_f_series = movie_features.loc[movie_id_for_explanation]

    combined_data = {}
    for col in original_X_cols:
        if col in user_f_series.index:
            combined_data[col] = user_f_series[col]
        elif col in movie_f_series.index:
            combined_data[col] = movie_f_series[col]
        else:
            combined_data[col] = 0 # Default for columns not in user or movie features

    # Ensure DataFrame has all original_X_columns in the correct order
    feature_vector = pd.DataFrame([combined_data], columns=original_X_cols).fillna(0)
    # Reorder columns to match X_train's column order explicitly
    feature_vector = feature_vector[original_X_cols]
    return feature_vector


def explain_xgb_recommendation_with_shap(feature_vector_df_shap, explainer_to_use, feature_names_list):
    if feature_vector_df_shap is None or feature_vector_df_shap.empty or explainer_to_use is None:
        print("SHAP: Invalid input (feature vector or explainer is None).")
        return None, None, None
    try:
        # Ensure feature_vector_df_shap columns are in the same order as feature_names_list if explainer relies on it
        # However, TreeExplainer typically works well if the DataFrame columns match training data columns
        shap_values_instance = explainer_to_use.shap_values(feature_vector_df_shap)

        # For single instance, single output regression, shap_values_instance is often a 1D array
        # explainer_to_use.expected_value should give the base value for TreeExplainer
        base_value = explainer_to_use.expected_value

        # Create a dictionary of feature_name: shap_value
        # shap_values_instance[0] because shap_values can be a list of arrays for multi-output models
        shap_values_dict = dict(zip(feature_names_list, shap_values_instance[0]))

        return shap_values_dict, base_value, shap_values_instance[0] # dict, base_value, raw_array
    except Exception as e:
        print(f"Error during SHAP explanation: {e}")
        return None, None, None

# Global xgb_model is used by lime_predict_fn_xgb
def lime_predict_fn_xgb(data_for_lime_np, columns_for_lime_list):
    # LIME passes a NumPy array; convert it to a DataFrame with correct column names and order
    df_for_prediction = pd.DataFrame(data_for_lime_np, columns=columns_for_lime_list)
    # Ensure the model sees columns in the exact order it was trained on (original_X_columns)
    # This step might be redundant if columns_for_lime_list is already original_X_columns
    # df_for_prediction = df_for_prediction[original_X_columns] # Assuming original_X_columns is accessible or passed
    return xgb_model.predict(df_for_prediction)


def explain_xgb_recommendation_with_lime(feature_vector_df_lime, explainer_to_use, num_lime_features=10):
    if feature_vector_df_lime is None or feature_vector_df_lime.empty or explainer_to_use is None:
        print("LIME: Invalid input (feature vector or explainer is None).")
        return None

    instance_to_explain_np = feature_vector_df_lime.iloc[0].values
    # Columns must match those used in lime_predict_fn_xgb and during explainer init
    lime_feature_names = feature_vector_df_lime.columns.tolist()

    # The predict_fn needs to be bound with the correct column names
    bound_predict_fn = lambda x: lime_predict_fn_xgb(x, lime_feature_names)

    try:
        explanation = explainer_to_use.explain_instance(
            data_row=instance_to_explain_np,
            predict_fn=bound_predict_fn,
            num_features=num_lime_features
        )
        return explanation.as_list() # Returns list of (feature_name, weight)
    except Exception as e:
        print(f"Error during LIME explanation: {e}")
        # import traceback
        # traceback.print_exc()
        return None


if __name__ == '__main__':
    # Initialize explainers after model (xgb_model) and training data (X_train) are ready.

    if 'xgb_model' in globals() and xgb_model is not None and \
       'X_train' in globals() and X_train is not None and not X_train.empty:

        print("\nInitializing SHAP Explainer...")
        # For TreeExplainer, providing background data (like X_train) is good practice.
        # It helps in more accurate estimation of expected values.
        shap_explainer_xgb = shap.TreeExplainer(xgb_model, X_train)
        print("SHAP TreeExplainer initialized.")

        print("\nInitializing LIME Explainer...")
        lime_explainer_xgb = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values, # LIME expects a NumPy array for training_data
            feature_names=X_train.columns.tolist(),
            class_names=['predicted_rating'], # Label for the output variable
            mode='regression',
            random_state=42,
            verbose=False # Can be set to True for more LIME output
        )
        print("LIME TabularExplainer initialized.")
    else:
        print("Warning: xgb_model or X_train not available. Explainers cannot be initialized.")
        shap_explainer_xgb = None
        lime_explainer_xgb = None

    print("\n\n--- New User Recommendations Test ---")
    new_user_input_ratings = [
        ("Freddy vs. Jason (2003)", 5.0),
        ("Shining, The (1980)", 5.0),
        ("Nightmare on Elm Street 2: Freddy's Revenge, A (1985)", 5.0),
        ("Pet Sematary (1989)", 5.0),
        ("Friday the 13th (2009)", 5.0),
        ("Toy Story (1995)", 3.0), # Mixed ratings
        ("Jumanji (1995)", 2.0),
        ("Mighty Morphin Power Rangers: The Movie (1995)", 1.0),
        ("Goofy Movie, A (1995)", 4.0),
        ("Wizard of Oz, The (1939)", 5.0)
    ]

    # Ensure all necessary arguments are passed to get_new_user_recommendations
    # original_X_columns should be X_train.columns (or X.columns if X_train is not yet defined at top level)
    # N_FACTORS_SVD should be accessible (defined globally)

    recommendations = get_new_user_recommendations(
        new_user_ratings_input=new_user_input_ratings,
        n=3, # Get top 3 recommendations
        model=xgb_model,
        all_movie_features_df=movie_features_df, # This is movie_features_df
        historical_user_factors_df=user_factors_df, # This is user_features_df containing SVD user factors
        original_X_columns=X_train.columns.tolist(), # Pass the column names list
        title_to_movie_id_map=title_to_movie_id,
        movie_id_to_title_map=movie_id_to_title,
        ratings_df=ratings_df_orig, # Use original ratings for finding similar users
        genre_columns=one_hot_genre_columns
    )

    if recommendations:
        print(f"\nTop {len(recommendations)} recommendations for the new user:")
        for i, rec in enumerate(recommendations):
            print(f"\n--- Recommendation {i+1} ---")
            print(f"Title: {rec['title']} (MovieID: {rec['movieId']})")
            print(f"Predicted Rating: {rec['predicted_rating']:.4f}")
            print(f"Uplift: {rec['uplift']:.4f}, Genre Similarity: {rec['genre_similarity']:.4f}, Adjusted Uplift: {rec['adjusted_uplift']:.4f}")

            if rec['shap_explanation']:
                print("\nSHAP Explanation (Top 5 positive contributors):")
                # Sort SHAP values by magnitude (absolute value) to show most impactful, or by value for direction
                sorted_shap = sorted(rec['shap_explanation'].items(), key=lambda item: item[1], reverse=True)
                for feature, shap_val in sorted_shap[:5]:
                    if shap_val > 0: # Only show positive contributions for brevity, or adjust as needed
                        print(f"  - {feature}: {shap_val:.4f}")
                print(f"  (SHAP Base Value: {rec['shap_base_value']:.4f})")
            else:
                print("\nSHAP Explanation: Not available.")

            if rec['lime_explanation']:
                print("\nLIME Explanation (Features and weights):")
                for feature, weight in rec['lime_explanation']:
                    print(f"  - {feature}: {weight:.4f}")
            else:
                print("\nLIME Explanation: Not available.")
        print("\n--- End of Recommendations ---")
    else:
        print("Could not generate recommendations for the new user or no recommendations met the criteria.")

    # Example of explaining a prediction for an *existing* user and a *specific* movie (if needed)
    # print("\n\n--- Existing User Explanation Test ---")
    # test_user_id_str = "1" # Example user ID
    # test_movie_id = 318 # Example movie ID (Shawshank Redemption)
    # if shap_explainer_xgb and lime_explainer_xgb and \
    #    test_user_id_str.isdigit() and int(test_user_id_str) in user_features_df.index and \
    #    test_movie_id in movie_features_df.index:

    #     print(f"Explaining recommendation for existing User ID: {test_user_id_str}, Movie ID: {test_movie_id} ({movie_id_to_title.get(test_movie_id)})")

    #     # Get the feature vector for this specific user-movie pair
    #     feature_vector_for_existing = get_feature_vector_for_existing_user_explanation(
    #         test_user_id_str,
    #         test_movie_id,
    #         user_features_df, # Full user features
    #         movie_features_df, # Full movie features
    #         X_train.columns.tolist() # original_X_columns
    #     )

    #     if feature_vector_for_existing is not None and not feature_vector_for_existing.empty:
    #         predicted_rating_existing = xgb_model.predict(feature_vector_for_existing)[0]
    #         print(f"Predicted rating for User {test_user_id_str} / Movie {test_movie_id}: {predicted_rating_existing:.4f}")

    #         shap_expl_existing, shap_base_existing, _ = explain_xgb_recommendation_with_shap(
    #             feature_vector_for_existing, shap_explainer_xgb, X_train.columns.tolist()
    #         )
    #         if shap_expl_existing:
    #             print("\nSHAP Explanation (Existing User - Top 5 positive):")
    #             sorted_shap_existing = sorted(shap_expl_existing.items(), key=lambda item: item[1], reverse=True)
    #             for feature, shap_val in sorted_shap_existing[:5]:
    #                  if shap_val > 0: print(f"  - {feature}: {shap_val:.4f}")
    #             print(f"  (SHAP Base Value: {shap_base_existing:.4f})")

    #         lime_expl_existing = explain_xgb_recommendation_with_lime(
    #             feature_vector_for_existing, lime_explainer_xgb, num_lime_features=10
    #         )
    #         if lime_expl_existing:
    #             print("\nLIME Explanation (Existing User):")
    #             for feature, weight in lime_expl_existing:
    #                 print(f"  - {feature}: {weight:.4f}")
    #     else:
    #         print(f"Could not generate feature vector for User {test_user_id_str}, Movie {test_movie_id}.")
    # else:
    #     print("\nSkipping existing user explanation test (explainers not ready or test IDs invalid).")

print("\nScript finished.")
