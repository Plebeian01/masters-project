# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:37:06 2025

This script builds and trains a movie recommendation model.
It preprocesses movie and user data, trains a RandomForestRegressor model,
and provides movie recommendations for new users.
"""
print("""\
This script performs the following actions:
1. Loads movie and rating data.
2. Engineers features for movies (genres, release year, popularity) and users (genre preferences, rating stats, activity, SVD latent factors).
3. Trains a RandomForestRegressor model to predict movie ratings.
4. Provides a function to recommend movies for new users based on seed ratings.
5. Uses SHAP to explain the model's recommendations for a given user and movie.
""")
"""
Created on Sun Jun 22 18:37:06 2025

@author: freez
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.sparse import csr_matrix

from sklearn.preprocessing   import StandardScaler, normalize
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error
import shap

# UMAP / PCA no longer needed here – we just train on the raw features!
# --- Paths ---
MOVIES_CSV  = "../dataset/ml-latest-small/movies.csv"
RATINGS_CSV = "../dataset/ml-latest-small/ratings.csv"

# --- Load Data ---
movies  = pd.read_csv(MOVIES_CSV)
ratings = pd.read_csv(RATINGS_CSV)
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# --- 1) Build movie_features ---
# 1a) genres one-hot
movie_genres = movies.set_index('movieId')['genres'] \
                     .str.get_dummies(sep='|')
# 1b) release year
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)
# 1c) popularity = # of ratings per movie
movie_popularity = ratings.groupby('movieId')['rating'] \
                          .count().rename('popularity')

movie_features = (
    movies[['movieId','year']]
    .set_index('movieId')
    .join(movie_genres)
    .join(movie_popularity, how='left')
    .fillna({'popularity':0})
)

movie_feature_cols = movie_features.columns.tolist()

# --- MERGE the same genre dummies into ratings so they exist for user‐aggregation! ---
ratings = ratings.merge(
    movie_genres.reset_index(),
    on='movieId',
    how='left'
).fillna(0)
gen_cols = movie_genres.columns.tolist()


# --- 2) Re-use your user_features pipeline ---
#    (genre sums/counts/means, entropy, stats, lifespan, frequency, latent SVD)

def genre_entropy(row):
    total = row.sum()
    if total <= 0:
        return 0.0
    p = row / total
    return entropy(p, base=2)

# filter active users
min_ratings_threshold = 20
active = ratings['userId'].value_counts()[lambda vc: vc>=min_ratings_threshold].index
ratings = ratings[ratings['userId'].isin(active)].copy()

# genre aggregations
gen_sum   = ratings.groupby('userId')[gen_cols].sum().add_suffix("_sum")
gen_mean  = ratings.groupby('userId')[gen_cols].mean().add_suffix("_mean")

# entropy
entropy_s = gen_mean.apply(genre_entropy, axis=1).rename("genre_entropy")

# rating stats
user_stats = ratings.groupby('userId')['rating'] \
                   .agg(avg_rating='mean',
                        rating_std='std',
                        rating_count='count')
user_stats['rating_std'].fillna(0, inplace=True)

# time features
ut = ratings.groupby('userId')['timestamp'].agg(first='min', last='max')
ut['lifespan_days']     = (ut['last'] - ut['first']).dt.days
ut['frequency_per_day'] = user_stats['rating_count'] / ut['lifespan_days'].replace(0,1)

# SVD latent factors
from sklearn.decomposition import TruncatedSVD
user_idx   = ratings['userId'].astype('category').cat.codes
movie_idx  = ratings['movieId'].astype('category').cat.codes
sparse_mtx = csr_matrix((ratings['rating'], (user_idx, movie_idx)))
# mean-center each row
lil = sparse_mtx.tolil()
for i,row in enumerate(lil.data):
    m = np.mean(row) if len(row)>0 else 0.0
    lil.data[i] = [v - m for v in row]
svd = TruncatedSVD(n_components=20, random_state=42)
latent = svd.fit_transform(lil.tocsr())
uids   = ratings['userId'].astype('category').cat.categories
latent_df = pd.DataFrame(latent,
                         index=uids,
                         columns=[f"latent_{i+1}" for i in range(latent.shape[1])])

# assemble user_features
user_features = pd.concat([
    gen_sum, gen_mean,
    entropy_s,
    user_stats,
    ut[['lifespan_days','frequency_per_day']],
    latent_df
], axis=1).fillna(0)
user_feature_cols = user_features.columns.tolist()


# --- 3) Build your training set: join user + movie features to each rating ---
ratings_train = ratings[['userId','movieId','rating']]

data = (
    ratings_train
      .merge(user_features,  left_on='userId',  right_index=True)
      .merge(movie_features, left_on='movieId', right_index=True)
)

data = data.replace([np.inf, -np.inf], np.nan)  # turn inf → nan
data = data.fillna(0.)                           # then nan → 0

X = data[user_feature_cols + movie_feature_cols]
y = data['rating']

# --- 4) Train / eval a RandomForestRegressor ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
rmse  = mean_squared_error(y_test, preds, squared=False)
print(f"Test RMSE = {rmse:.3f}")


# --- 5) Recommendation function for a brand-new user ---
def recommend(new_user_ratings, top_k=10):
    """
    new_user_ratings: Dict[movieId -> rating]
    Returns: (recommendation DataFrame, X_new feature-frame)
    """
    # build a little DataFrame of their seed ratings
    df = pd.DataFrame.from_dict(new_user_ratings, orient='index', columns=['rating']) \
          .merge(movie_features, left_index=True, right_index=True, how='left') \
          .fillna(0)

    # per-user genre stats
    gsum   = df[gen_cols].mul(df['rating'], axis=0).sum()
    gcount = df[gen_cols].count()
    gmean  = gsum.div(gcount.replace(0,1))
    ent    = genre_entropy(gmean)
    avg_r  = np.mean(list(new_user_ratings.values()))
    cnt    = len(new_user_ratings)

    # build the one-row user_features vector
    user_row = pd.Series(
        {**gsum.add_suffix('_sum').to_dict(),
         **gmean.add_suffix('_mean').to_dict(),
         'genre_entropy':        ent,
         'avg_rating':           avg_r,
         'rating_std':           np.std(list(new_user_ratings.values())),
         'rating_count':         cnt,
         'lifespan_days':        0,
         'frequency_per_day':    0,
         **{c:0 for c in latent_df.columns}
        },
        name='new_user'
    ).reindex(user_feature_cols).fillna(0)

      # 5b) assemble one row per candidate movie
    all_movies = movie_features.index
    uf = pd.DataFrame([user_row.values]*len(all_movies),
                      index=all_movies, columns=user_feature_cols)
    mf = movie_features.loc[all_movies, movie_feature_cols]
    X_new = pd.concat([uf, mf], axis=1)

    # <<< NEW: scrub X_new just like training data! >>>
    X_new = X_new.replace([np.inf, -np.inf], np.nan)
    X_new = X_new.fillna(0.)

    # 5c) predict & take top-K
    preds = rf.predict(X_new)
    rec   = (
        pd.Series(preds, index=X_new.index, name='pred_rating')
          .nlargest(top_k)
          .reset_index()
          .merge(movies[['movieId','title']], on='movieId')
    )
    return rec, X_new


# example usage:
seed = {1:5.0, 50:4.0, 110:5.0}
top5, X_new = recommend(seed, top_k=5)
print(top5)


# --- 6) Explain with SHAP ---
explainer = shap.TreeExplainer(rf)
to_explain = X_new.loc[[ top5.movieId.iloc[0] ]]
shap_vals   = explainer.shap_values(to_explain)

shap.initjs()
shap.force_plot(
    explainer.expected_value, shap_vals, to_explain,
    feature_names=X_new.columns.tolist()
)