"""
Model building and evaluation utilities for Amazon Food Recommender.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from src.amazon_recommender.config import CFG
from src.amazon_recommender.logging_setup import logger

def build_sparse_matrix(df, user_col='UserId', item_col='ProductId', rating_col='Score'):
    user_index = {u:i for i,u in enumerate(sorted(df[user_col].unique()))}
    item_index = {p:i for i,p in enumerate(sorted(df[item_col].unique()))}
    rows = df[user_col].map(user_index)
    cols = df[item_col].map(item_index)
    data = df[rating_col].values
    logger.info(f"Building sparse matrix with shape ({len(user_index)}, {len(item_index)})")
    return coo_matrix((data, (rows, cols)), shape=(len(user_index), len(item_index))).tocsr(), user_index, item_index

def build_recommender(train_matrix, n_components=None, n_neighbors=None):
    if n_components is None:
        n_components = CFG["model"]["svd_components"]
    if n_neighbors is None:
        n_neighbors = CFG["model"]["knn_neighbors"]
    logger.info(f"Training SVD with n_components={n_components}")
    n_features = train_matrix.shape[1]
    svd = TruncatedSVD(n_components=min(50, n_features), random_state=42)
    features = svd.fit_transform(train_matrix)
    n_samples = train_matrix.shape[0]
    actual_n_neighbors = min(n_neighbors, n_samples)
    logger.info(f"Training KNN with n_neighbors={actual_n_neighbors}")
    knn = NearestNeighbors(n_neighbors=actual_n_neighbors, metric='cosine', algorithm='brute')
    knn.fit(features)
    return svd, knn, features
