import pandas as pd
import pickle
import numpy as np
from src.amazon_recommender.model import build_sparse_matrix, build_recommender

class Recommender:
    def __init__(self, data_path, knn_model_path, svd_model_path):
        # Load data & build train matrix
        df = pd.read_csv(data_path)
        self.train_matrix, self.user_index, self.item_index = build_sparse_matrix(df)

        # Load pre-trained models (or None)
        with open(svd_model_path, "rb") as f:
            loaded_svd = pickle.load(f)
        with open(knn_model_path, "rb") as f:
            loaded_knn = pickle.load(f)

        if loaded_svd is None or loaded_knn is None:
            # Fallback: train a fresh pipeline
            self.svd, self.knn, self.features = build_recommender(self.train_matrix)
        else:
            self.svd = loaded_svd
            self.knn = loaded_knn
            # Compute features for KNN
            self.features = self.svd.transform(self.train_matrix)

    def top_n(self, user_id, k=10):
        # Return empty list if user is unknown
        if user_id not in self.user_index:
            return []
        idx = self.user_index[user_id]
        # Find neighbors in latent space
        _, neigh_idxs = self.knn.kneighbors(
            [self.features[idx]], n_neighbors=self.knn.n_neighbors
        )
        # Average their rows to get item scores
        rec_scores = self.train_matrix[neigh_idxs.flatten(), :].mean(axis=0)
        # Pick top‑k items
        top_indices = np.array(rec_scores).flatten().argsort()[::-1][:k]
        # Map back to original item IDs
        inv_item_index = {v: k for k, v in self.item_index.items()}
        return [inv_item_index[i] for i in top_indices]
