# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:37:06 2025

@author: freez
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
# from surprise.accuracy import rmse # For SVD accuracy if needed
from collections import defaultdict
import shap
import lime
import lime.lime_tabular
import xgboost as xgb # Added XGBoost
from sklearn.model_selection import train_test_split as sklearn_train_test_split # For XGBoost
from sklearn.metrics import mean_squared_error

# --- 0. Initial Data Loading (from original plan) ---
print("Step 0: Loading and Basic Preprocessing Data...")

# Load datasets
movies_df = pd.read_csv('../dataset/100kDataset/movies.csv')
ratings_df = pd.read_csv('../dataset/100kDataset/ratings.csv')

# Display basic info
# print("Movies DataFrame:")
# movies_df.info()
# print("\nRatings DataFrame:")
# ratings_df.info()

print("\nMovies head:")
print(movies_df.head())
print("\nRatings head:")
print(ratings_df.head())

# Merge dataframes
df = pd.merge(ratings_df, movies_df, on='movieId')

# Extract year from title
df['year'] = df['title'].str.extract(r'\((\d{4})\)')
df['year'] = pd.to_numeric(df['year'], errors='coerce') # errors='coerce' will turn non-numeric years into NaT/NaN

# Feature Engineering: One-hot encode genres
genres_dummies = df['genres'].str.get_dummies(sep='|')
df = pd.concat([df, genres_dummies], axis=1)
print("\nDataFrame with genres one-hot encoded (first few rows):")
print(df.head())

# Check for missing values after transformations
# print("\nMissing values in merged DataFrame after feature engineering:")
# print(df.isnull().sum())

# For SVD, we primarily need user, item, and rating.
reader = Reader(rating_scale=(0.5, 5.0))
surprise_data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader) # Renamed to surprise_data

print("\nData loaded into Surprise Dataset format.")
print("Basic preprocessing complete.")


# --- Plan Step 1: SVD Feature Generation & Initial Setup (New Plan) ---
print("\nStep 1: SVD Feature Generation & Initial Setup...")

N_FACTORS_SVD = 50  # Number of latent factors for SVD features
svd_model = SVD(n_factors=N_FACTORS_SVD, n_epochs=20, random_state=42, verbose=False)

# Build the full trainset from the surprise_data
full_trainset_svd = surprise_data.build_full_trainset() # Explicitly named for SVD

print("Training SVD model on the full dataset to extract latent factors...")
svd_model.fit(full_trainset_svd)
print("SVD model training complete.")

# Extract user latent factors
user_factors_list = []
for inner_uid in full_trainset_svd.all_users():
    raw_uid = full_trainset_svd.to_raw_uid(inner_uid)
    factors = svd_model.pu[inner_uid]
    user_factors_list.append([raw_uid] + factors.tolist())

user_factors_df = pd.DataFrame(user_factors_list, columns=['userId'] + [f'uf_svd_{i}' for i in range(N_FACTORS_SVD)])
user_factors_df.set_index('userId', inplace=True) # Set userId as index
print(f"\nExtracted {len(user_factors_df)} user SVD factors.")
print("User SVD factors (user_factors_df) head:")
print(user_factors_df.head())

# Extract item latent factors
item_factors_list = []
for inner_iid in full_trainset_svd.all_items():
    raw_iid = full_trainset_svd.to_raw_iid(inner_iid)
    factors = svd_model.qi[inner_iid]
    item_factors_list.append([raw_iid] + factors.tolist())

item_factors_df = pd.DataFrame(item_factors_list, columns=['movieId'] + [f'if_svd_{i}' for i in range(N_FACTORS_SVD)])
item_factors_df.set_index('movieId', inplace=True) # Set movieId as index
print(f"\nExtracted {len(item_factors_df)} item SVD factors.")
print("Item SVD factors (item_factors_df) head:")
print(item_factors_df.head())

# --- Mappings (retained for convenience) ---
print("\nSetting up movie title-ID mappings...")
movie_id_to_title = movies_df.set_index('movieId')['title'].to_dict()
title_to_movie_id = {title: id for id, title in movie_id_to_title.items()}
print("Mappings complete.")


# --- Plan Step 2: Extended Feature Engineering ---
print("\nStep 2: Extended Feature Engineering...")

# User-level features
print("Calculating user-level features (average rating, number of ratings)...")
user_stats = ratings_df.groupby('userId')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'user_avg_rating', 'count': 'user_num_ratings'})
# Merge with SVD user factors
user_features_df = user_factors_df.merge(user_stats, on='userId', how='left')
# Fill NaNs for users who might be in SVD factors but somehow not in ratings_df (should be rare if SVD trained on full data)
user_features_df.fillna({'user_avg_rating': full_trainset_svd.global_mean, 'user_num_ratings': 0}, inplace=True)

print("User features (user_features_df) head:")
print(user_features_df.head())

# Movie-level features
print("\nCalculating movie-level features (average rating, number of ratings)...")
movie_stats = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).rename(columns={'mean': 'movie_avg_rating', 'count': 'movie_num_ratings'})

# Base movie features from original movies_df (movieId, title, year, genres one-hot)
# We need 'year' and the one-hot encoded genre columns.
# `df` already contains `movieId`, `year` (extracted), and one-hot encoded genres.

# Identify one-hot genre columns dynamically from `df`
# These are columns that are not the primary ones and not SVD factors or calculated stats.
# A simpler way: they are the columns from `genres_dummies`.
one_hot_genre_columns = genres_dummies.columns.tolist()

# Select 'movieId', 'year', and one-hot genre columns from the main 'df', then drop duplicates
# and set 'movieId' as index.
base_movie_meta_features = df[['movieId', 'year'] + one_hot_genre_columns].drop_duplicates(subset=['movieId'])
base_movie_meta_features = base_movie_meta_features.set_index('movieId')


# Merge movie_stats with SVD item factors
movie_features_df = item_factors_df.merge(movie_stats, on='movieId', how='left')
# Merge with base movie meta features (year, genres)
movie_features_df = movie_features_df.merge(base_movie_meta_features, on='movieId', how='left')

# Fill NaNs for movies
movie_features_df.fillna({'movie_avg_rating': full_trainset_svd.global_mean,
                          'movie_num_ratings': 0}, inplace=True)
# For genre columns, NaN means the movie doesn't have that genre, so fill with 0.
# These are now part of movie_features_df from the merge with base_movie_meta_features
movie_features_df[one_hot_genre_columns] = movie_features_df[one_hot_genre_columns].fillna(0)

# For 'year', fill with median year.
if 'year' in movie_features_df.columns: # 'year' should be present
    # Addressing FutureWarning for pandas 3.0
    movie_features_df['year'] = movie_features_df['year'].fillna(movie_features_df['year'].median())
else:
    print("Warning: 'year' column not found in movie_features_df before NaN fill.")


print("\nMovie features (movie_features_df) head:")
print(movie_features_df.head())
print("\nMovie features columns:")
print(movie_features_df.columns)


# --- Plan Step 3: Construct XGBoost Training Set ---
print("\nStep 3: Constructing XGBoost Training Set...")

# Merge ratings_df with user_features_df and movie_features_df
xgb_train_df = ratings_df.merge(user_features_df, on='userId', how='left')
xgb_train_df = xgb_train_df.merge(movie_features_df, on='movieId', how='left')

print(f"Shape of merged dataframe for XGBoost: {xgb_train_df.shape}")
print("Merged dataframe for XGBoost (xgb_train_df) head:")
print(xgb_train_df.head())

# Define target variable y
y = xgb_train_df['rating']

# Define features X - drop identifiers, raw genre string, and target
# Also drop 'title' from movies_df if it got pulled in, and 'timestamp'
columns_to_drop_for_X = ['userId', 'movieId', 'rating', 'timestamp', 'title', 'genres']
# Check if 'title' and 'genres' (original string) are present before dropping
if 'title' not in xgb_train_df.columns: # title might not be in xgb_train_df if not merged from movies_df directly
    columns_to_drop_for_X.remove('title')
if 'genres' not in xgb_train_df.columns: # genres string might not be in xgb_train_df
    columns_to_drop_for_X.remove('genres')


X = xgb_train_df.drop(columns=columns_to_drop_for_X, errors='ignore')

# Sanity check for NaNs in X before training
print(f"\nNumber of NaNs in X before final fill: {X.isnull().sum().sum()}")
# Fill any remaining NaNs in feature set (e.g., with 0 or median)
# For simplicity, filling with 0. This might not be optimal for all features.
# Year was filled with median, SVD factors shouldn't be NaN if merges are correct.
# Genre dummies were filled with 0. User/Movie stats NaNs were handled.
X.fillna(0, inplace=True)
print(f"Number of NaNs in X after final fill: {X.isnull().sum().sum()}")


# Ensure all feature names are strings (XGBoost requirement)
X.columns = [str(col) for col in X.columns]

print("\nFeatures X for XGBoost (first 5 rows):")
print(X.head())
print("\nShape of X:", X.shape)
print("Target y for XGBoost (first 5 values):")
print(y.head())
print("Shape of y:", y.shape)


# Split data into training and testing sets for XGBoost
X_train, X_test, y_train, y_test = sklearn_train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("XGBoost training set construction complete.")


# --- Plan Step 4: Train XGBoost Regressor ---
print("\nStep 4: Training XGBoost Regressor...")

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression task
    n_estimators=100,              # Number of trees (can be tuned)
    learning_rate=0.1,             # Learning rate (can be tuned)
    max_depth=5,                   # Max depth of a tree (can be tuned)
    subsample=0.8,                 # Subsample ratio of the training instance
    colsample_bytree=0.8,          # Subsample ratio of columns when constructing each tree
    random_state=42,
    n_jobs=-1                      # Use all available cores
)

print("Fitting XGBoost model...")
xgb_model.fit(X_train, y_train)
print("XGBoost model training complete.")

# Evaluate the model on the test set
print("\nEvaluating XGBoost model on the test set...")
y_pred_test = xgb_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"XGBoost Test RMSE: {rmse_test:.4f}")


# --- Plan Step 5: Implement Recommendation Generation with XGBoost ---
print("\nStep 5: Implementing Recommendation Generation with XGBoost...")

MIN_RATINGS_THRESHOLD = 10 # Default minimum ratings threshold

def get_xgb_recommendations(user_id_str, n, model, user_features, all_movie_features, svd_trainset, original_X_columns, min_ratings_threshold=MIN_RATINGS_THRESHOLD):
    """
    Generates movie recommendations for a user using the trained XGBoost model.
    Args:
        user_id_str (str): The raw user ID.
        n (int): Number of recommendations to return.
        model (xgb.XGBRegressor): The trained XGBoost model.
        user_features (pd.DataFrame): DataFrame containing all user features, indexed by userId.
        all_movie_features (pd.DataFrame): DataFrame containing all movie features, indexed by movieId.
        svd_trainset (surprise.Trainset): The SVD trainset to find movies already rated by the user.
        original_X_columns (pd.Index): The columns from the training data X, in the correct order.
        min_ratings_threshold (int): Minimum number of ratings a movie must have to be recommended.
    Returns:
        list: A list of dictionaries, where each dictionary contains 'title', 'predicted_rating', and 'movieId'.
    """
    user_id = int(user_id_str) # Convert to int for lookups if necessary

    if user_id not in user_features.index:
        print(f"User ID {user_id} not found in user_features_df. Cannot generate recommendations.")
        return []

    user_f = user_features.loc[user_id]

    # Get movies already rated by the user from the SVD trainset
    try:
        user_inner_id = svd_trainset.to_inner_uid(user_id)
        rated_movie_raw_ids = {svd_trainset.to_raw_iid(item_inner_id) for item_inner_id, _ in svd_trainset.ur[user_inner_id]}
    except ValueError: # User not in SVD trainset (e.g. new user not in original ratings)
        print(f"User ID {user_id} not in SVD trainset. Assuming no movies rated for exclusion.")
        rated_movie_raw_ids = set()

    candidate_movies = []
    all_movie_ids_in_features = all_movie_features.index

    for movie_id in all_movie_ids_in_features:
        if movie_id not in rated_movie_raw_ids:
            if movie_id in all_movie_features.index:
                movie_f = all_movie_features.loc[movie_id]

                # Apply minimum ratings threshold
                if movie_f.get('movie_num_ratings', 0) < min_ratings_threshold:
                    continue

                # Combine user and movie features
                # The order of features must match X_train.columns
                # User features first, then movie features as per typical df merge user_df.merge(movie_df)
                # This depends on how X was constructed.
                # Our X was: ratings_df.merge(user_features, on='userId').merge(movie_features, on='movieId')
                # So, user features (excluding SVD factors if they were separate) come first, then movie features.
                # Let's reconstruct the feature vector carefully based on original_X_columns.

                # Create a combined series/dict for current user-movie pair
                # SVD user factors are in user_f, SVD item factors are in movie_f.
                # Other user stats (user_avg_rating, etc.) are in user_f.
                # Other movie stats (movie_avg_rating, year, genres) are in movie_f.

                # Start with an empty series with the correct index (original_X_columns)
                combined_features_data = {}

                # Populate from user_f and movie_f
                for col in original_X_columns:
                    if col in user_f.index:
                        combined_features_data[col] = user_f[col]
                    elif col in movie_f.index:
                        combined_features_data[col] = movie_f[col]
                    else:
                        # This case should ideally not happen if feature engineering is consistent
                        # and all columns in original_X_columns are found in either user_f or movie_f
                        # print(f"Warning: Column {col} not found in user_f or movie_f. Filling with 0.")
                        combined_features_data[col] = 0 # Default fill

                # Convert to DataFrame for prediction
                feature_vector_df = pd.DataFrame([combined_features_data], columns=original_X_columns)
                feature_vector_df = feature_vector_df.fillna(0) # Final safety net for NaNs

                try:
                    predicted_rating = model.predict(feature_vector_df)[0]
                    candidate_movies.append({'movieId': movie_id, 'predicted_rating': predicted_rating})
                except Exception as e:
                    print(f"Error predicting for movie {movie_id}: {e}")
                    print(f"Feature vector causing error:\n{feature_vector_df}")
                    continue # Skip this movie
            # else: # Movie not in all_movie_features (should not happen if all_movie_ids_in_features is from its index)
            #    print(f"MovieId {movie_id} not found in all_movie_features during candidate generation.")


    # Sort movies by predicted rating
    candidate_movies.sort(key=lambda x: x['predicted_rating'], reverse=True)

    top_n_recommendations = []
    for rec in candidate_movies[:n]:
        top_n_recommendations.append({
            'title': movie_id_to_title.get(rec['movieId'], "Unknown Movie"),
            'predicted_rating': rec['predicted_rating'],
            'movieId': rec['movieId']
        })

    return top_n_recommendations

print("XGBoost recommendation function `get_xgb_recommendations` defined.")


# Sections for old SVD recommendation (get_recommendations_for_user) and
# old LIME/SHAP for SVD are intentionally removed here.
# They will be replaced by XGBoost specific functions later in the plan.


if __name__ == '__main__':
    print("\n--- Hybrid Movie Recommendation System (SVD + XGBoost) ---")
    print("Step 1: SVD Feature Generation complete.")
    print("Step 2: Extended Feature Engineering complete.")
    print("Step 3: XGBoost Training Set construction complete.")
    print("Step 4: XGBoost Regressor training complete.")
    print(f"   XGBoost Test RMSE: {rmse_test:.4f}")
    print("Step 5: XGBoost Recommendation Generation function defined.")

    # Example: Print number of users and items from SVD model perspective
    print(f"\nNumber of users in SVD trainset: {full_trainset_svd.n_users}")
    print(f"Number of items (movies) in SVD trainset: {full_trainset_svd.n_items}")
    print(f"Shape of user_features_df: {user_features_df.shape}")
    print(f"Shape of movie_features_df: {movie_features_df.shape}")
    print(f"Shape of X_train for XGBoost: {X_train.shape}")

    # Example usage of the new recommendation function
    example_user_id_xgb = '1'
    print(f"\n--- Example: XGBoost Recommendations for User ID {example_user_id_xgb} ---")
    # Ensure X_train.columns is available in this scope if running __main__ directly after defining X_train
    # It should be, as X_train is defined globally in the script flow before __main__
    xgb_recs = get_xgb_recommendations(example_user_id_xgb, 5, xgb_model, user_features_df, movie_features_df, full_trainset_svd, X_train.columns) # min_ratings_threshold will use default

    if xgb_recs:
        print(f"\nTop 5 recommendations for user {example_user_id_xgb} using XGBoost (filtered by min_ratings_threshold={MIN_RATINGS_THRESHOLD}):")
        for rec in xgb_recs:
            print(f"- {rec['title']} (Predicted XGBoost Rating: {rec['predicted_rating']:.4f}) (MovieID: {rec['movieId']})")
    else:
        print(f"Could not generate XGBoost recommendations for user {example_user_id_xgb} (possibly due to filtering).")


    # Placeholder for subsequent steps (XGBoost training, recs, explanations)
    print("\nNext steps will involve: Explanation for XGBoost (SHAP, LIME)...")


# --- Plan Step 6: Implement Explainability for XGBoost (SHAP & LIME) ---
print("\nStep 6: Implementing Explainability for XGBoost...")

# SHAP Explainer
# For tree models like XGBoost, TreeExplainer is efficient.
# It needs the model and optionally the training data for background distribution (can improve consistency).
# Using X_train for background data is a good practice.
shap_explainer_xgb = shap.TreeExplainer(xgb_model, data=X_train) # Pass X_train as background

def get_feature_vector_for_explanation(user_id_str, movie_id_for_explanation, user_features_df, movie_features_df, original_X_columns):
    """
    Helper function to construct the exact feature vector for a given user-movie pair,
    matching the structure of X_train.
    """
    user_id = int(user_id_str)
    if user_id not in user_features_df.index or movie_id_for_explanation not in movie_features_df.index:
        print("User or Movie ID not found in feature dataframes for explanation.")
        return None

    user_f = user_features_df.loc[user_id]
    movie_f = movie_features_df.loc[movie_id_for_explanation]

    combined_features_data = {}
    for col in original_X_columns: # original_X_columns is X_train.columns
        if col in user_f.index:
            combined_features_data[col] = user_f[col]
        elif col in movie_f.index:
            combined_features_data[col] = movie_f[col]
        else:
            combined_features_data[col] = 0 # Should match filling strategy for X_train

    feature_vector_df = pd.DataFrame([combined_features_data], columns=original_X_columns)
    feature_vector_df = feature_vector_df.fillna(0) # Ensure consistency
    return feature_vector_df


def explain_xgb_recommendation_with_shap(feature_vector_df_shap, shap_explainer_to_use):
    """
    Explains an XGBoost recommendation using SHAP.
    Args:
        feature_vector_df_shap (pd.DataFrame): Single row DataFrame of the instance to explain.
        shap_explainer_to_use (shap.TreeExplainer): The SHAP TreeExplainer initialized with the model.
    Returns:
        tuple: (shap_values, expected_value) for the instance.
               shap_values is an array of SHAP values for each feature.
    """
    if feature_vector_df_shap is None or feature_vector_df_shap.empty:
        return None, None

    # SHAP values for a single instance
    shap_values_instance = shap_explainer_to_use.shap_values(feature_vector_df_shap)
    # For single output regression, shap_values_instance is a 1D array for the first (and only) instance.
    # If feature_vector_df_shap has one row, shap_values_instance[0] gives feature importances for that row.

    return shap_values_instance[0], shap_explainer_to_use.expected_value


# LIME Explainer
# LIME needs training data to understand feature distributions.
lime_explainer_xgb = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values, # Pass numpy array
    feature_names=X_train.columns.tolist(),
    class_names=['predicted_rating'],
    mode='regression',
    random_state=42
)

def lime_predict_fn_xgb(data_for_lime):
    """
    Prediction function for LIME, compatible with XGBoost.
    LIME passes a NumPy array, so convert it to DataFrame with correct column names.
    """
    return xgb_model.predict(pd.DataFrame(data_for_lime, columns=X_train.columns))

def explain_xgb_recommendation_with_lime(feature_vector_df_lime, lime_explainer_to_use, predict_fn_for_lime, num_lime_features=10):
    """
    Explains an XGBoost recommendation using LIME.
    Args:
        feature_vector_df_lime (pd.DataFrame): Single row DataFrame of the instance to explain.
        lime_explainer_to_use (lime.lime_tabular.LimeTabularExplainer): The LIME explainer.
        predict_fn_for_lime (callable): The prediction function for LIME.
        num_lime_features (int): Number of features to show in LIME explanation.
    Returns:
        lime.explanation.Explanation: The LIME explanation object.
    """
    if feature_vector_df_lime is None or feature_vector_df_lime.empty:
        return None

    # LIME expects a 1D numpy array for the instance to explain
    instance_to_explain_lime = feature_vector_df_lime.iloc[0].values

    explanation = lime_explainer_to_use.explain_instance(
        data_row=instance_to_explain_lime,
        predict_fn=predict_fn_for_lime,
        num_features=num_lime_features
    )
    return explanation

print("SHAP and LIME explanation functions for XGBoost defined.")


if __name__ == '__main__':
    print("\n--- Hybrid Movie Recommendation System (SVD + XGBoost) ---")
    print("Step 1: SVD Feature Generation complete.")
    print("Step 2: Extended Feature Engineering complete.")
    print("Step 3: XGBoost Training Set construction complete.")
    print("Step 4: XGBoost Regressor training complete.")
    print(f"   XGBoost Test RMSE: {rmse_test:.4f}")
    print("Step 5: XGBoost Recommendation Generation function defined.")
    print("Step 6: SHAP and LIME explanation functions for XGBoost defined.")

    # Example: Print number of users and items from SVD model perspective
    print(f"\nNumber of users in SVD trainset: {full_trainset_svd.n_users}")
    print(f"Number of items (movies) in SVD trainset: {full_trainset_svd.n_items}")
    print(f"Shape of user_features_df: {user_features_df.shape}")
    print(f"Shape of movie_features_df: {movie_features_df.shape}")
    print(f"Shape of X_train for XGBoost: {X_train.shape}")

    # Example usage of the new recommendation function
    example_user_id_xgb = '1'
    print(f"\n--- Example: XGBoost Recommendations for User ID {example_user_id_xgb} ---")
    # Passing the threshold explicitly to show it can be changed, or rely on default
    xgb_recs = get_xgb_recommendations(example_user_id_xgb, 5, xgb_model, user_features_df, movie_features_df, full_trainset_svd, X_train.columns, min_ratings_threshold=MIN_RATINGS_THRESHOLD)


    if xgb_recs:
        print(f"\nTop 5 recommendations for user {example_user_id_xgb} using XGBoost (filtered by min_ratings_threshold={MIN_RATINGS_THRESHOLD}):")
        for rec in xgb_recs:
            print(f"- {rec['title']} (Predicted XGBoost Rating: {rec['predicted_rating']:.4f}) (MovieID: {rec['movieId']})")

        # Explain the first recommendation
        if xgb_recs: # Check again, in case filtering resulted in no recs
            first_rec_movie_id_xgb = xgb_recs[0]['movieId']
            first_rec_title_xgb = xgb_recs[0]['title']
            predicted_rating_xgb = xgb_recs[0]['predicted_rating']

            print(f"\n--- Explaining recommendation for '{first_rec_title_xgb}' (Rating: {predicted_rating_xgb:.4f}) to User {example_user_id_xgb} ---")

            # Get the feature vector for this specific recommendation
            instance_feature_vector_df = get_feature_vector_for_explanation(example_user_id_xgb, first_rec_movie_id_xgb, user_features_df, movie_features_df, X_train.columns)

            if instance_feature_vector_df is not None:
                # SHAP Explanation
                print("\nSHAP Explanation:")
                shap_values_instance, shap_expected_value = explain_xgb_recommendation_with_shap(instance_feature_vector_df, shap_explainer_xgb)
                if shap_values_instance is not None:
                    print(f"  Base value (expected XGBoost output): {shap_expected_value:.4f}")
                    print(f"  Current prediction: {predicted_rating_xgb:.4f} (SHAP sum: {shap_expected_value + np.sum(shap_values_instance):.4f})") # Verify sum

                    feature_names_for_shap = X_train.columns.tolist()
                    shap_contributions = sorted(list(zip(feature_names_for_shap, shap_values_instance)), key=lambda x: abs(x[1]), reverse=True)
                    print("  Top 5 contributing features (SHAP):")
                    for feature, shap_val in shap_contributions[:5]:
                        print(f"    {feature}: {shap_val:.4f}")
                else:
                    print("  Could not generate SHAP explanation.")

                # LIME Explanation
                print("\nLIME Explanation:")
                lime_explanation_xgb = explain_xgb_recommendation_with_lime(instance_feature_vector_df, lime_explainer_xgb, lime_predict_fn_xgb, num_lime_features=5)
                if lime_explanation_xgb:
                    print("  Top 5 contributing features (LIME):")
                    for feature_name, weight in lime_explanation_xgb.as_list():
                        print(f"    {feature_name}: {weight:.4f}")
                else:
                    print("  Could not generate LIME explanation.")
            else:
                print("Could not retrieve feature vector for explanation.")
    else:
        print(f"Could not generate XGBoost recommendations for user {example_user_id_xgb} (possibly due to filtering by min_ratings_threshold={MIN_RATINGS_THRESHOLD}).")


    print("\n--- End of XGBoost Hybrid Recommendation System Script ---")


# --- Plan Step 1 & 2 (New Plan for New User): Define and Integrate `get_new_user_recommendations` Function ---
# print("\nStep 7: Defining function for new user recommendations...") # Already part of the function log

def get_new_user_recommendations(
    new_user_ratings_input, # List of tuples: [('Movie Title 1', 5.0), ...]
    n,
    model,
    all_movie_features_df,
    historical_user_factors_df, # This is user_factors_df
    original_X_columns,
    title_to_movie_id_map,
    movie_id_to_title_map,
    min_ratings_threshold=MIN_RATINGS_THRESHOLD # Added threshold
    ):
    """
    Generates movie recommendations for a new user based on a small list of liked movies.
    SVD factors for the new user are estimated using the average of historical users.
    Filters movies by minimum number of ratings.
    """
    print(f"\nGenerating recommendations for a new user who liked: {new_user_ratings_input}")

    # Convert input movie titles to movieIds and collect ratings
    liked_movie_ids = set()
    valid_ratings = []
    for title, rating in new_user_ratings_input:
        movie_id = title_to_movie_id_map.get(title)
        if movie_id:
            liked_movie_ids.add(movie_id)
            valid_ratings.append(rating)
        else:
            print(f"Warning: Movie title '{title}' not found in dataset. Skipping for new user profile.")

    if not liked_movie_ids:
        print("No valid liked movies found for the new user. Cannot generate recommendations.")
        return []

    # Calculate new user's explicit features
    new_user_avg_rating = np.mean(valid_ratings) if valid_ratings else 0.0
    new_user_num_ratings = len(valid_ratings)

    print(f"New user profile: Avg Rating={new_user_avg_rating:.2f}, Num Ratings={new_user_num_ratings}")

    # Calculate new user's SVD factors (average of historical users)
    avg_user_svd_factors = historical_user_factors_df[[col for col in historical_user_factors_df.columns if col.startswith('uf_svd_')]].mean()

    new_user_feature_data = {
        'user_avg_rating': new_user_avg_rating,
        'user_num_ratings': new_user_num_ratings
    }
    for col, val in avg_user_svd_factors.items():
        if col in original_X_columns: # Ensure only relevant SVD factors are added
             new_user_feature_data[col] = val

    candidate_movies_predictions = []
    for movie_id_candidate in all_movie_features_df.index:
        if movie_id_candidate not in liked_movie_ids:
            movie_f_candidate = all_movie_features_df.loc[movie_id_candidate]

            # Apply minimum ratings threshold for new user recommendations
            if movie_f_candidate.get('movie_num_ratings', 0) < min_ratings_threshold:
                continue

            combined_features_for_pred = {}
            for col_template in original_X_columns:
                if col_template in new_user_feature_data: # User-specific features first
                    combined_features_for_pred[col_template] = new_user_feature_data[col_template]
                elif col_template in movie_f_candidate.index: # Then movie features
                    combined_features_for_pred[col_template] = movie_f_candidate[col_template]
                else: # Fallback for any missing columns (should be rare if data is clean)
                    combined_features_for_pred[col_template] = 0

            feature_vector_df_pred = pd.DataFrame([combined_features_for_pred], columns=original_X_columns)
            feature_vector_df_pred = feature_vector_df_pred.fillna(0) # Ensure no NaNs for prediction

            try:
                predicted_rating = model.predict(feature_vector_df_pred)[0]
                candidate_movies_predictions.append({'movieId': movie_id_candidate, 'predicted_rating': predicted_rating})
            except Exception as e:
                print(f"Error predicting for movie {movie_id_candidate} for new user: {e}")
                continue

    candidate_movies_predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)

    top_n_recommendations = []
    for rec in candidate_movies_predictions[:n]:
        top_n_recommendations.append({
            'title': movie_id_to_title_map.get(rec['movieId'], "Unknown Movie"),
            'predicted_rating': rec['predicted_rating'],
            'movieId': rec['movieId']
        })

    return top_n_recommendations

# Moved the print statement here to reflect it's part of the script execution flow now.
print("Function `get_new_user_recommendations` defined (Step 7).")


if __name__ == '__main__':
    # ... (previous __main__ content remains unchanged up to historical user explanations)

    # --- Test new user recommendation ---
    print("\n\n--- Testing New User Recommendation Scenario ---")
    new_user_liked_movies = [
        ("Toy Story (1995)", 5.0),
        ("Jumanji (1995)", 5.0),
        ("Mighty Morphin Power Rangers: The Movie (1995)", 5.0), # Less popular, might be filtered
        ("Goofy Movie, A (1995)", 5.0), # Less popular
        ("Wizard of Oz, The (1939)", 5.0) # Popular
    ]

    # Using the default MIN_RATINGS_THRESHOLD for new user recs
    new_user_recs = get_new_user_recommendations(
        new_user_ratings_input=new_user_liked_movies,
        n=5,
        model=xgb_model,
        all_movie_features_df=movie_features_df,
        historical_user_factors_df=user_factors_df,
        original_X_columns=X_train.columns,
        title_to_movie_id_map=title_to_movie_id,
        movie_id_to_title_map=movie_id_to_title
        # min_ratings_threshold will use default from function definition
    )

    if new_user_recs:
        print(f"\nTop 5 recommendations for the new user (filtered by min_ratings_threshold={MIN_RATINGS_THRESHOLD}):")
        for rec in new_user_recs:
            print(f"- {rec['title']} (Predicted XGBoost Rating: {rec['predicted_rating']:.4f}) (MovieID: {rec['movieId']})")
    else:
        print(f"Could not generate recommendations for the new user (possibly due to filtering by min_ratings_threshold={MIN_RATINGS_THRESHOLD}).")

print("\n--- End of Script ---") # Simplified end message
