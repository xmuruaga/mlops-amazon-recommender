""" Serving and recommendation utilities for Amazon Food Recommender. """
import numpy as np
import pandas as pd

def recommend(user_id, train_matrix, svd, knn, features, user_index, k=10):
    if user_id not in user_index:
        return []
    idx = user_index[user_id]
    _, neigh_idxs = knn.kneighbors([features[idx]], n_neighbors=knn.n_neighbors)
    neighs = [list(user_index.keys())[i] for i in neigh_idxs.flatten() if i < len(user_index)]
    recs = train_matrix[neigh_idxs.flatten()].mean(axis=0)
    top_items = np.array(recs).flatten().argsort()[::-1][:k]
    return top_items
"""
Serving and recommendation utilities for Amazon Food Recommender.
"""
import numpy as np
import pandas as pd

def recommend(user_id, train_matrix, svd, knn, features, user_index, k=10):
    if user_id not in user_index:
        return []
    idx = user_index[user_id]
    _, neigh_idxs = knn.kneighbors([features[idx]], n_neighbors=knn.n_neighbors)
    neighs = [list(user_index.keys())[i] for i in neigh_idxs.flatten() if i < len(user_index)]
    recs = train_matrix[neigh_idxs.flatten()].mean(axis=0)
    top_items = np.array(recs).flatten().argsort()[::-1][:k]
    return top_items
