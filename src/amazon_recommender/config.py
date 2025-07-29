import yaml
import os

def load(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Override with environment variables if present
    cfg["dataset"]["n_users"] = int(os.getenv("N_USERS", cfg["dataset"]["n_users"]))
    cfg["dataset"]["n_items"] = int(os.getenv("N_ITEMS", cfg["dataset"]["n_items"]))
    cfg["model"]["svd_components"] = int(os.getenv("SVD_COMPONENTS", cfg["model"]["svd_components"]))
    cfg["model"]["knn_neighbors"] = int(os.getenv("KNN_NEIGHBORS", cfg["model"]["knn_neighbors"]))
    return cfg

CFG = load()
import yaml
import os

def load(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Override with environment variables if present
    cfg["dataset"]["n_users"] = int(os.getenv("N_USERS", cfg["dataset"]["n_users"]))
    cfg["dataset"]["n_items"] = int(os.getenv("N_ITEMS", cfg["dataset"]["n_items"]))
    cfg["model"]["svd_components"] = int(os.getenv("SVD_COMPONENTS", cfg["model"]["svd_components"]))
    cfg["model"]["knn_neighbors"] = int(os.getenv("KNN_NEIGHBORS", cfg["model"]["knn_neighbors"]))
    return cfg

CFG = load()
