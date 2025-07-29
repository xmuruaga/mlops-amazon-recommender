import os
import pickle
import random
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, generate_latest
from pydantic import BaseModel, HttpUrl

# ——— Configuration & Artifact Loading ———

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

# Load models
with open(os.path.join(ARTIFACTS_DIR, "svd_model.pkl"), "rb") as f:
    svd_model = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "knn_model.pkl"), "rb") as f:
    knn_model = pickle.load(f)

# Load user–item matrix (likely normalized)
train_matrix = pd.read_pickle(os.path.join(ARTIFACTS_DIR, "train_matrix.pkl"))

# Optional product metadata
product_info_path = os.path.join(ARTIFACTS_DIR, "product_info.pkl")
if os.path.exists(product_info_path):
    with open(product_info_path, "rb") as f:
        product_info = pickle.load(f)
else:
    product_info = {}

# ——— Raw Reviews & True Averages ———

REVIEWS_CSV = os.path.join(BASE_DIR, "data", "raw", "Reviews.csv")
if os.path.exists(REVIEWS_CSV):
    raw_reviews = pd.read_csv(REVIEWS_CSV, low_memory=False)
    # compute true per‐ASIN mean scores
    avg_ratings_map = (
        raw_reviews
        .groupby("ProductId")["Score"]
        .mean()
        .to_dict()
    )
    # we don’t need the full DataFrame beyond this point
    del raw_reviews
else:
    avg_ratings_map = {}

# ——— App & Metrics ———

app = FastAPI(title="Amazon Fine Food Recommender API")
recommend_counter = Counter("recommend_requests_total", "Total recommend requests")

# ——— Pydantic Schemas ———

class Recommendation(BaseModel):
    asin: str
    product_url: HttpUrl
    thumbnail_url: HttpUrl
    title: str = ""
    score: float
    avg_rating: Optional[float] = None

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]

# ——— Core Recommendation Logic ———

def recommend(user_id: str, train_mat: pd.DataFrame, knn, features, k: int = 10):
    if user_id not in train_mat.index:
        return []
    idx = train_mat.index.get_loc(user_id)
    _, neigh_idxs = knn.kneighbors([features[idx]], n_neighbors=knn.n_neighbors)
    neighs = [train_mat.index[i] for i in neigh_idxs.flatten() if i < len(train_mat)]
    recs = (
        train_mat
        .loc[neighs]
        .mean()
        .sort_values(ascending=False)
        .head(k)
        .index
        .tolist()
    )
    return recs

# ——— API Endpoints ———

@app.get("/api/users")
def get_user_ids(n: int = Query(10, description="How many random user IDs to return")):
    """
    Return a random sample of up to `n` user IDs
    from the training matrix (or fall back to Reviews.csv).
    """
    try:
        all_ids = list(train_matrix.index)
    except Exception:
        # fallback to raw CSV
        if os.path.exists(REVIEWS_CSV):
            df = pd.read_csv(REVIEWS_CSV, low_memory=False)
            all_ids = df["UserId"].unique().tolist()
        else:
            all_ids = []

    sample_size = min(n, len(all_ids))
    sampled = random.sample(all_ids, sample_size) if sample_size > 0 else []
    return JSONResponse(content={"user_ids": sampled})


@app.get("/user_stats")
def get_user_stats(user_id: str, n: int = 10):
    """
    For a given user_id, show total ratings, mean score,
    and last n purchases (with title, URL, user and avg score).
    """
    # try raw CSV first
    if os.path.exists(REVIEWS_CSV):
        df = pd.read_csv(REVIEWS_CSV, low_memory=False)
        if "Time" in df.columns:
            df["ReviewTime"] = pd.to_datetime(df["Time"], unit="s")
        user_reviews = df[df["UserId"] == user_id]
        num_ratings = len(user_reviews)
        avg_rating = float(user_reviews["Score"].mean()) if num_ratings else None

        # sort to get latest purchases
        if "ReviewTime" in user_reviews.columns:
            last = user_reviews.sort_values("ReviewTime", ascending=False).head(n)
        else:
            last = user_reviews.head(n)

        purchases = []
        for _, row in last.iterrows():
            pid = row["ProductId"]
            purchases.append({
                "product_id": pid,
                "title": row.get("Summary", ""),
                "product_url": f"https://www.amazon.com/dp/{pid}",
                "user_rating": float(row["Score"]),
                "avg_rating": avg_ratings_map.get(pid)
            })

        return JSONResponse({
            "num_ratings": num_ratings,
            "avg_rating": avg_rating,
            "last_purchases": purchases
        })

    # fallback to train_matrix only
    if user_id not in train_matrix.index:
        return JSONResponse({"num_ratings": 0, "avg_rating": None, "last_purchases": []})

    user_ratings = train_matrix.loc[user_id].dropna()
    num_ratings = user_ratings.size
    avg_rating = float(user_ratings.mean()) if num_ratings else None
    top = user_ratings.sort_values(ascending=False).head(n)

    purchases = []
    for asin, usr_score in top.items():
        purchases.append({
            "product_id": asin,
            "title": product_info.get(asin, {}).get("title", ""),
            "product_url": f"https://www.amazon.com/dp/{asin}",
            "user_rating": float(usr_score),
            "avg_rating": avg_ratings_map.get(asin)
        })

    return JSONResponse({
        "num_ratings": num_ratings,
        "avg_rating": avg_rating,
        "last_purchases": purchases
    })


@app.get("/recommend", response_model=RecommendResponse)
def get_recommendations(
    user_id: str = Query(..., description="User ID"),
    k: int = Query(10, description="Number of recommendations")
):
    """
    Return the top‐k neighbor‐based recommendations for a given user,
    with true average product ratings.
    """
    recommend_counter.inc()
    # Get the SVD‐feature matrix once per request
    features = svd_model.transform(train_matrix.values)
    rec_asins = recommend(user_id, train_matrix, knn_model, features, k)

    recommendations: List[Recommendation] = []
    for asin in rec_asins:
        # neighbor‐based “score” (from your train_matrix means)
        neigh_scores = train_matrix[asin].dropna()
        score = float(neigh_scores.mean()) if not neigh_scores.empty else 0.0

        # true average from raw reviews
        avg_rating = avg_ratings_map.get(asin)

        recommendations.append(Recommendation(
            asin=asin,
            product_url=f"https://www.amazon.com/dp/{asin}",
            thumbnail_url=f"https://images.amazon.com/images/P/{asin}",
            title=product_info.get(asin, {}).get("title", ""),
            score=score,
            avg_rating=avg_rating
        ))

    return RecommendResponse(user_id=user_id, recommendations=recommendations)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
