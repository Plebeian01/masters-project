# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:54:28 2025

@author: freez
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP
import matplotlib.pyplot as plt

# --- Paths ---
MOVIES_CSV  = "../dataset/ml-latest-small/movies.csv"
RATINGS_CSV = "../dataset/ml-latest-small/ratings.csv"

# --- Load Data ---
movies  = pd.read_csv(MOVIES_CSV)
ratings = pd.read_csv(RATINGS_CSV)
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# --- Extract release year & movie popularity ---
# assumes titles end in "(YYYY)"
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')[0].astype(float)
movie_popularity = (
    ratings
    .groupby('movieId')['rating']
    .count()
    .rename('popularity')
)
ratings = (
    ratings
    .merge(movies[['movieId','year']], on='movieId', how='left')
    .merge(movie_popularity,        on='movieId', how='left')
)
ratings['popularity'].fillna(0, inplace=True)

# --- Filter active users ---
min_ratings_threshold = 20
user_counts = ratings['userId'].value_counts()
active_users = user_counts[user_counts >= min_ratings_threshold].index
ratings = ratings[ratings['userId'].isin(active_users)].copy()

# --- Genre features ---
genres   = movies['genres'].str.get_dummies(sep='|')
gen_cols = genres.columns.tolist()
ratings  = ratings.merge(
    movies[['movieId']].join(genres),
    on='movieId', how='left'
)

# --- Genre aggregations ---
gen_sum   = ratings.groupby('userId')[gen_cols].sum().rename(
    columns=lambda g: f'{g}_sum'
)
gen_count = ratings.groupby('userId')[gen_cols].count().rename(
    columns=lambda g: f'{g}_count'
)
gen_mean  = ratings.groupby('userId')[gen_cols].mean().rename(
    columns=lambda g: f'{g}_mean'
)

# --- Genre entropy ---
def genre_entropy(means):
    s = means.sum()
    if s <= 0:
        return 0.0
    p = means / s
    return entropy(p, base=2)

entropy_series = (
    ratings
    .groupby('userId')[gen_cols]
    .mean()
    .apply(genre_entropy, axis=1)
    .rename('genre_entropy')
)

# --- Rating behavior stats ---
user_stats = ratings.groupby('userId')['rating'].agg(
    avg_rating='mean',
    rating_std ='std',
    rating_count='count'
)
user_stats['rating_std'].fillna(0, inplace=True)

# --- Account lifespan & frequency ---
user_time = ratings.groupby('userId')['timestamp'].agg(first='min', last='max')
user_time['lifespan_days']     = (user_time['last'] - user_time['first']).dt.days
user_time['frequency_per_day'] = (
    user_stats['rating_count'] /
    user_time['lifespan_days'].replace(0,1)
)

# --- Build a single categoricals object to keep userID ↔ row aligned ---
user_cat  = ratings['userId'].astype('category')
user_idx  = user_cat.cat.codes
user_ids  = user_cat.cat.categories    # <-- same ordering as user_idx
movie_idx = ratings['movieId'].astype('category').cat.codes

# --- Raw sparse user–item matrix ---
raw_matrix = csr_matrix(
    (ratings['rating'].values, (user_idx, movie_idx))
)

# --- Binary “liked” mask + TF–IDF + L2-normalize ---
liked_mask = (raw_matrix >= 4).astype(int)
tfidf      = TfidfTransformer(norm=None)
user_tfidf = tfidf.fit_transform(liked_mask)
user_norm  = normalize(user_tfidf, norm='l2', axis=1)

# --- Per-user avg liked-year & liked-popularity ---
liked_ratings = ratings[ratings['rating'] >= 4]
user_year = (
    liked_ratings
    .groupby('userId')['year']
    .mean()
    .reindex(user_ids)
    .fillna(0.0)
    .rename('avg_liked_year')
)
user_pop = (
    liked_ratings
    .groupby('userId')['popularity']
    .mean()
    .reindex(user_ids)
    .fillna(0.0)
    .rename('avg_liked_popularity')
)

# --- Stack numeric features, down-weight them, and standardize ---
num_feats = np.vstack([user_year.values, user_pop.values]).T
scaler_num = StandardScaler()
num_feats_scaled = scaler_num.fit_transform(num_feats)

alpha = 0.3   # down-weight factor; tweak between 0.1–1.0
num_feats_scaled *= alpha

# --- Combine dense L2 TF–IDF + numeric feats ---
user_array = user_norm.toarray()
features_full = np.hstack([user_array, num_feats_scaled])

# --- Dimensionality reduction: PCA → UMAP ---
pca = PCA(n_components=50, random_state=42)
pca_emb = pca.fit_transform(features_full)

umap_model = UMAP(
    n_components=10,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
features = umap_model.fit_transform(pca_emb)   # <— this is your array

# --- Silhouette sweep to pick k (look for the “next” peak after k=2) ---
sil_scores = []
k_range    = range(2, 16)
for kk in k_range:
    km_   = KMeans(n_clusters=kk, init='k-means++', random_state=42)
    preds = km_.fit_predict(features)
    sil   = silhouette_score(features, preds, metric='cosine')
    sil_scores.append(sil)

# plot
plt.figure(figsize=(8,4))
plt.plot(list(k_range), sil_scores, 'o-')
plt.xlabel('k'); plt.ylabel('Silhouette (cosine)')
plt.title('Silhouette Analysis')
plt.grid(True)
plt.show()

# --- Choose your k_final as a local peak (e.g. 5) rather than blind k=2 ---
k_final = 7
km = KMeans(n_clusters=k_final, init='k-means++', random_state=42)
clusters = km.fit_predict(features)

print(f"Cosine silhouette (k={k_final}):",
      silhouette_score(features, clusters, metric='cosine').round(3))

# --- Map back & produce per-cluster top-N movies ---
cluster_map = pd.Series(clusters, index=user_ids, name='cluster')
ratings_with_cluster = ratings.merge(
    cluster_map.rename('cluster'),
    left_on='userId', right_index=True
)

cluster_movie_stats = (
    ratings_with_cluster
    .groupby(['cluster','movieId'])
    .agg(
      avg_rating   = ('rating','mean'),
      rating_count = ('rating','count')
    )
    .reset_index()
)

top_n = 10
min_ratings = 5
for c in range(k_final):
    sub = cluster_movie_stats[
        (cluster_movie_stats['cluster']    == c) &
        (cluster_movie_stats['rating_count']>=min_ratings)
    ].sort_values(['avg_rating','rating_count'], ascending=False)

    top = (sub
           .head(top_n)
           .merge(movies[['movieId','title']], on='movieId', how='left')
           [['movieId','title','avg_rating','rating_count']]
    )
    print(f"\n=== Cluster {c} top {top_n} ===")
    if top.empty:
        print("  (no movies with ≥ {min_ratings} ratings in this cluster)")
    else:
        print(top.to_string(index=False))

# --- 2D UMAP for visualization ---
umap_vis = UMAP(
    n_components=2, n_neighbors=10, min_dist=0.3,
    metric='cosine', random_state=42
)
vis_2d = umap_vis.fit_transform(pca_emb)  # visualize on PCA→UMAP pipeline

viz_df = pd.DataFrame(vis_2d, index=user_ids, columns=['UMAP1','UMAP2'])
viz_df['Cluster'] = clusters

plt.figure(figsize=(10,6))
sc = plt.scatter(
    viz_df['UMAP1'], viz_df['UMAP2'],
    c=viz_df['Cluster'], cmap='tab20', s=10, alpha=0.6
)
plt.colorbar(sc, label='Cluster ID')
plt.title('User Clusters (UMAP 2D)')
plt.xlabel('UMAP1'); plt.ylabel('UMAP2')
plt.grid(True); plt.tight_layout()
plt.show()
















