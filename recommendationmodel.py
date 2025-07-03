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
from sklearn.metrics.pairwise import cosine_similarity


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
ABS_PRED_RATING_THRESHOLD = 4.0 # Default absolute predicted rating threshold

def get_xgb_recommendations(user_id_str, n, model, user_features, all_movie_features, svd_trainset, original_X_columns, 
                            min_ratings_threshold=MIN_RATINGS_THRESHOLD, 
                            abs_pred_rating_threshold=ABS_PRED_RATING_THRESHOLD):
    """
    Generates movie recommendations for a user using the trained XGBoost model,
    based on uplift and absolute predicted rating.
    Args:
        user_id_str (str): The raw user ID.
        n (int): Number of recommendations to return.
        model (xgb.XGBRegressor): The trained XGBoost model.
        user_features (pd.DataFrame): DataFrame containing all user features, indexed by userId.
        all_movie_features (pd.DataFrame): DataFrame containing all movie features, indexed by movieId.
        svd_trainset (surprise.Trainset): The SVD trainset to find movies already rated by the user.
        original_X_columns (pd.Index): The columns from the training data X, in the correct order.
        min_ratings_threshold (int): Minimum number of ratings a movie must have to be considered.
        abs_pred_rating_threshold (float): Minimum predicted rating for the user for a movie to be considered.
    Returns:
        list: A list of dictionaries, where each dictionary contains 'title', 'predicted_rating', 'uplift', and 'movieId'.
    """
    user_id = int(user_id_str) # Convert to int for lookups if necessary

    if user_id not in user_features.index:
        print(f"User ID {user_id} not found in user_features_df. Cannot generate recommendations.")
        return []

    user_f = user_features.loc[user_id]

    try:
        user_inner_id = svd_trainset.to_inner_uid(user_id)
        rated_movie_raw_ids = {svd_trainset.to_raw_iid(item_inner_id) for item_inner_id, _ in svd_trainset.ur[user_inner_id]}
    except ValueError:
        print(f"User ID {user_id} not in SVD trainset. Assuming no movies rated for exclusion.")
        rated_movie_raw_ids = set()

    candidate_movies = []
    all_movie_ids_in_features = all_movie_features.index

    for movie_id in all_movie_ids_in_features:
        if movie_id not in rated_movie_raw_ids:
            if movie_id in all_movie_features.index:
                movie_f = all_movie_features.loc[movie_id]

                if movie_f.get('movie_num_ratings', 0) < min_ratings_threshold:
                    continue

                combined_features_data = {}
                for col in original_X_columns:
                    if col in user_f.index:
                        combined_features_data[col] = user_f[col]
                    elif col in movie_f.index:
                        combined_features_data[col] = movie_f[col]
                    else:
                        combined_features_data[col] = 0 

                feature_vector_df = pd.DataFrame([combined_features_data], columns=original_X_columns)
                feature_vector_df = feature_vector_df.fillna(0)

                try:
                    predicted_rating = model.predict(feature_vector_df)[0]
                    
                    if predicted_rating >= abs_pred_rating_threshold:
                        movie_avg_rating = movie_f.get('movie_avg_rating', full_trainset_svd.global_mean) # Use global mean if somehow missing
                        uplift = predicted_rating - movie_avg_rating
                        candidate_movies.append({
                            'movieId': movie_id, 
                            'predicted_rating': predicted_rating,
                            'uplift': uplift
                        })
                except Exception as e:
                    print(f"Error predicting for movie {movie_id}: {e}")
                    print(f"Feature vector causing error:\n{feature_vector_df}")
                    continue

    # Sort movies by uplift
    candidate_movies.sort(key=lambda x: x['uplift'], reverse=True)

    top_n_recommendations = []
    for rec in candidate_movies[:n]:
        top_n_recommendations.append({
            'title': movie_id_to_title.get(rec['movieId'], "Unknown Movie"),
            'predicted_rating': rec['predicted_rating'],
            'uplift': rec['uplift'],
            'movieId': rec['movieId']
        })

    return top_n_recommendations

print("XGBoost recommendation function `get_xgb_recommendations` defined.")


# Sections for old SVD recommendation (get_recommendations_for_user) and
# old LIME/SHAP for SVD are intentionally removed here.

if __name__ == '__main__':
    print("\n--- Hybrid Movie Recommendation System (SVD + XGBoost) ---")
    print("Step 1: SVD Feature Generation complete.")
    print("Step 2: Extended Feature Engineering complete.")
    print("Step 3: XGBoost Training Set construction complete.")
    print("Step 4: XGBoost Regressor training complete.")
    print(f"   XGBoost Test RMSE: {rmse_test:.4f}")
    print("Step 5: XGBoost Recommendation Generation function defined.")

    print(f"\nNumber of users in SVD trainset: {full_trainset_svd.n_users}")
    print(f"Number of items (movies) in SVD trainset: {full_trainset_svd.n_items}")
    print(f"Shape of user_features_df: {user_features_df.shape}")
    print(f"Shape of movie_features_df: {movie_features_df.shape}")
    print(f"Shape of X_train for XGBoost: {X_train.shape}")

    example_user_id_xgb = '1'
    print(f"\n--- Example: XGBoost Recommendations for User ID {example_user_id_xgb} (Uplift-based) ---")
    
    xgb_recs = get_xgb_recommendations(
        example_user_id_xgb, 
        5, 
        xgb_model, 
        user_features_df, 
        movie_features_df, 
        full_trainset_svd, 
        X_train.columns
        # min_ratings_threshold and abs_pred_rating_threshold will use defaults
    )

    if xgb_recs:
        print(f"\nTop 5 recommendations for user {example_user_id_xgb} based on Uplift (min_ratings={MIN_RATINGS_THRESHOLD}, abs_pred_rating>={ABS_PRED_RATING_THRESHOLD}):")
        for rec in xgb_recs:
            print(f"- {rec['title']} (Predicted Rating: {rec['predicted_rating']:.4f}, Uplift: {rec['uplift']:.4f}) (MovieID: {rec['movieId']})")
    else:
        print(f"Could not generate XGBoost recommendations for user {example_user_id_xgb} (possibly due to filtering).")

    print("\nNext steps will involve: Explanation for XGBoost (SHAP, LIME)...")


# --- Plan Step 6: Implement Explainability for XGBoost (SHAP & LIME) ---
print("\nStep 6: Implementing Explainability for XGBoost...")

shap_explainer_xgb = shap.TreeExplainer(xgb_model, data=X_train) 

def get_feature_vector_for_explanation(user_id_str, movie_id_for_explanation, user_features_df, movie_features_df, original_X_columns):
    user_id = int(user_id_str)
    if user_id not in user_features_df.index or movie_id_for_explanation not in movie_features_df.index:
        print("User or Movie ID not found in feature dataframes for explanation.")
        return None

    user_f = user_features_df.loc[user_id]
    movie_f = movie_features_df.loc[movie_id_for_explanation]

    combined_features_data = {}
    for col in original_X_columns: 
        if col in user_f.index:
            combined_features_data[col] = user_f[col]
        elif col in movie_f.index:
            combined_features_data[col] = movie_f[col]
        else:
            combined_features_data[col] = 0 

    feature_vector_df = pd.DataFrame([combined_features_data], columns=original_X_columns)
    feature_vector_df = feature_vector_df.fillna(0) 
    return feature_vector_df


def explain_xgb_recommendation_with_shap(feature_vector_df_shap, shap_explainer_to_use):
    if feature_vector_df_shap is None or feature_vector_df_shap.empty:
        return None, None
    shap_values_instance = shap_explainer_to_use.shap_values(feature_vector_df_shap)
    return shap_values_instance[0], shap_explainer_to_use.expected_value

lime_explainer_xgb = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values, 
    feature_names=X_train.columns.tolist(),
    class_names=['predicted_rating'],
    mode='regression',
    random_state=42
)

def lime_predict_fn_xgb(data_for_lime):
    return xgb_model.predict(pd.DataFrame(data_for_lime, columns=X_train.columns))

def explain_xgb_recommendation_with_lime(feature_vector_df_lime, lime_explainer_to_use, predict_fn_for_lime, num_lime_features=10):
    if feature_vector_df_lime is None or feature_vector_df_lime.empty:
        return None
    instance_to_explain_lime = feature_vector_df_lime.iloc[0].values
    explanation = lime_explainer_to_use.explain_instance(
        data_row=instance_to_explain_lime,
        predict_fn=predict_fn_for_lime,
        num_features=num_lime_features
    )
    return explanation

print("SHAP and LIME explanation functions for XGBoost defined.")


if __name__ == '__main__':
    # This block will re-run with the updated get_xgb_recommendations if the script is executed.
    # The previous print statements for __main__ are sufficient.
    # For SHAP/LIME, we'd typically explain one of the new uplift-based recommendations.

    if xgb_recs: # If recommendations were generated
        first_rec_movie_id_xgb = xgb_recs[0]['movieId']
        first_rec_title_xgb = xgb_recs[0]['title']
        predicted_rating_xgb = xgb_recs[0]['predicted_rating']
        uplift_xgb = xgb_recs[0]['uplift']

        print(f"\n--- Explaining recommendation for '{first_rec_title_xgb}' (Pred Rating: {predicted_rating_xgb:.4f}, Uplift: {uplift_xgb:.4f}) to User {example_user_id_xgb} ---")

        instance_feature_vector_df = get_feature_vector_for_explanation(example_user_id_xgb, first_rec_movie_id_xgb, user_features_df, movie_features_df, X_train.columns)

        if instance_feature_vector_df is not None:
            print("\nSHAP Explanation:")
            shap_values_instance, shap_expected_value = explain_xgb_recommendation_with_shap(instance_feature_vector_df, shap_explainer_xgb)
            if shap_values_instance is not None:
                print(f"  Base value (expected XGBoost output): {shap_expected_value:.4f}")
                print(f"  Current prediction: {predicted_rating_xgb:.4f} (SHAP sum: {shap_expected_value + np.sum(shap_values_instance):.4f})")

                feature_names_for_shap = X_train.columns.tolist()
                shap_contributions = sorted(list(zip(feature_names_for_shap, shap_values_instance)), key=lambda x: abs(x[1]), reverse=True)
                print("  Top 5 contributing features (SHAP):")
                for feature, shap_val in shap_contributions[:5]:
                    print(f"    {feature}: {shap_val:.4f}")
            else:
                print("  Could not generate SHAP explanation.")

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


    print("\n--- End of XGBoost Hybrid Recommendation System Script ---")


# --- Plan Step 1 & 2 (New Plan for New User): Define and Integrate `get_new_user_recommendations` Function ---
print("Function `get_new_user_recommendations` defined (Step 7).") # Placeholder, will be updated in next step

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
    Hybrid recommendation function for new users using similar users' SVD + genre profile similarity.
    """
    print(f"\nGenerating recommendations for a new user who liked: {new_user_ratings_input}")

    # --- Step 1: Convert movie titles to IDs ---
    liked_movie_ids = set()
    valid_ratings = []
    for title, rating in new_user_ratings_input:
        movie_id = title_to_movie_id_map.get(title)
        if movie_id:
            liked_movie_ids.add(movie_id)
            valid_ratings.append(rating)
        else:
            print(f"Warning: '{title}' not found in dataset. Skipping.")

    if not liked_movie_ids:
        print("No valid liked movies found. Aborting.")
        return []

    new_user_avg_rating = np.mean(valid_ratings) if valid_ratings else 0.0 
    new_user_num_ratings = len(valid_ratings)

    # --- Step 2: Estimate SVD latent factors using similar users ---
    def find_similar_users(liked_movie_ids, ratings_df, min_rating=4.0):
        users = set()
        for movie_id in liked_movie_ids:
            matched = ratings_df[(ratings_df['movieId'] == movie_id) & (ratings_df['rating'] >= min_rating)]
            users.update(matched['userId'].tolist())
        return list(users)

    similar_users = find_similar_users(liked_movie_ids, ratings_df, min_rating=min_similar_rating)
    if len(similar_users) < min_similar_users:
        print(f"Only {len(similar_users)} similar users found. Using global SVD mean.")
        estimated_user_factors = historical_user_factors_df.mean()
    else:
        valid_similar_users_df = historical_user_factors_df.loc[historical_user_factors_df.index.intersection(similar_users)]
        estimated_user_factors = valid_similar_users_df.mean()
        print(f"Estimated user profile from {len(valid_similar_users_df)} similar users.")

    # --- Step 3: Compute genre profile of liked movies ---
    def compute_genre_profile(liked_movie_ids, movie_features_df, genre_columns):
        liked_df = movie_features_df.loc[movie_features_df.index.intersection(liked_movie_ids)]
        return liked_df[genre_columns].mean().values.reshape(1, -1)

    user_genre_vector = compute_genre_profile(liked_movie_ids, all_movie_features_df, genre_columns)

    # --- Step 4: Assemble user feature vector ---
    new_user_feature_data = {
        'user_avg_rating': new_user_avg_rating,
        'user_num_ratings': new_user_num_ratings
    }
    for col, val in estimated_user_factors.items():
        if col in original_X_columns:
            new_user_feature_data[col] = val

    # --- Step 5: Score all candidate movies ---
    candidate_movies_predictions = []
    for movie_id_candidate in all_movie_features_df.index:
        if movie_id_candidate not in liked_movie_ids:
            movie_f = all_movie_features_df.loc[movie_id_candidate]
            if movie_f.get('movie_num_ratings', 0) < min_ratings_threshold:
                continue

            # Cosine similarity of genres
            movie_genre_vector = movie_f[genre_columns].values.reshape(1, -1)
            genre_similarity = cosine_similarity(user_genre_vector, movie_genre_vector)[0][0]

            if genre_similarity < genre_similarity_threshold:
                continue  # Skip weak matches

            combined_features = {}
            for col in original_X_columns:
                if col in new_user_feature_data:
                    combined_features[col] = new_user_feature_data[col]
                elif col in movie_f.index:
                    combined_features[col] = movie_f[col]
                else:
                    combined_features[col] = 0

            feature_vector_df = pd.DataFrame([combined_features], columns=original_X_columns).fillna(0)

            try:
                predicted_rating = model.predict(feature_vector_df)[0]
                if predicted_rating >= abs_pred_rating_threshold:
                    movie_avg_rating = movie_f.get('movie_avg_rating', 3.5)
                    uplift = predicted_rating - movie_avg_rating
                    adjusted_uplift = uplift * genre_similarity

                    candidate_movies_predictions.append({
                        'movieId': movie_id_candidate,
                        'predicted_rating': predicted_rating,
                        'uplift': uplift,
                        'adjusted_uplift': adjusted_uplift,
                        'genre_similarity': genre_similarity
                    })
            except Exception as e:
                print(f"Error for movie {movie_id_candidate}: {e}")
                continue

    # --- Step 6: Return Top-N Recommendations ---
    candidate_movies_predictions.sort(key=lambda x: x['adjusted_uplift'], reverse=True)

    top_n = [
        {
            'title': movie_id_to_title_map.get(rec['movieId'], "Unknown Movie"),
            'predicted_rating': rec['predicted_rating'],
            'uplift': rec['uplift'],
            'genre_similarity': rec['genre_similarity'],
            'adjusted_uplift': rec['adjusted_uplift'],
            'movieId': rec['movieId']
        }
        for rec in candidate_movies_predictions[:n]
    ]

    return top_n


if __name__ == '__main__':
    # --- Test new user recommendation (Uplift-based) ---
    print("\n\n--- Testing New User Recommendation Scenario (Uplift-based) ---")
    new_user_liked_movies = [
        ("Toy Story (1995)", 5.0),
        ("Jumanji (1995)", 5.0),
        ("Mighty Morphin Power Rangers: The Movie (1995)", 5.0), 
        ("Goofy Movie, A (1995)", 5.0), 
        ("Wizard of Oz, The (1939)", 5.0)
    ]

    new_user_recs = get_new_user_recommendations(
        new_user_ratings_input=new_user_liked_movies,
        n=5,
        model=xgb_model,
        all_movie_features_df=movie_features_df,
        historical_user_factors_df=user_factors_df,
        original_X_columns=X_train.columns,
        title_to_movie_id_map=title_to_movie_id,
        movie_id_to_title_map=movie_id_to_title,
        ratings_df=ratings_df,
        genre_columns=one_hot_genre_columns  
)

    if new_user_recs:
        print(f"\nTop 5 recommendations for the new user based on Uplift (min_ratings={MIN_RATINGS_THRESHOLD}, abs_pred_rating>={ABS_PRED_RATING_THRESHOLD}):")
        for rec in new_user_recs:
            print(f"- {rec['title']} (Predicted Rating: {rec['predicted_rating']:.4f}, Uplift: {rec['uplift']:.4f}) (MovieID: {rec['movieId']})")
    else:
        print(f"Could not generate recommendations for the new user (possibly due to filtering).")

print("\n--- End of Script ---")
