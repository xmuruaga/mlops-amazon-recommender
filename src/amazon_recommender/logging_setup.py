# src/amazon_recommender/logging_setup.py
import logging

# configure root logger (or pick your own name)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("amazon_recommender")
