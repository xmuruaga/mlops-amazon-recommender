"""
Data loading and cleaning utilities for Amazon Food Recommender.
"""
import os
import pandas as pd
from src.amazon_recommender.config import CFG
from src.amazon_recommender.logging_setup import logger

def download_reviews():
    """Download Reviews.csv from Kaggle CLI if not already present."""
    raw_dir = os.path.dirname(os.path.abspath(CFG["paths"]["raw_csv"]))
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath(os.path.join('..', '.kaggle'))
    os.makedirs(raw_dir, exist_ok=True)
    csv_path = os.path.abspath(CFG["paths"]["raw_csv"])
    if not os.path.exists(csv_path):
        import subprocess
        try:
            subprocess.run([
                'kaggle', 'datasets', 'download',
                '-d', 'snap/amazon-fine-food-reviews',
                '-p', raw_dir, '--unzip'
            ], check=True)
            logger.info(f"Downloaded Reviews.csv to {csv_path}")
        except subprocess.CalledProcessError:
            logger.error("Failed to download Reviews.csv. Have you accepted the dataset rules on Kaggle?")
            raise RuntimeError("Failed to download Reviews.csv. Have you accepted the dataset rules on Kaggle?")
    else:
        logger.info("Reviews.csv already exists. Skipping download.")
    return csv_path

def load_and_clean(csv_path: str = None) -> pd.DataFrame:
    if csv_path is None:
        csv_path = os.path.abspath(CFG["paths"]["raw_csv"])
    df = pd.read_csv(csv_path)
    logger.info(f"Initial shape: {df.shape}")
    # Drop rows with null UserId
    df = df[df["UserId"].notnull()]
    # Deduplicate user-product pairs
    df = df.drop_duplicates(subset=["UserId", "ProductId"], keep="first")
    logger.info(f"Shape after cleaning: {df.shape}")
    return df
