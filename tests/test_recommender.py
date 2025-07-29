import pytest
from src.amazon_recommender.models.recommender import Recommender
import pandas as pd
import pickle

@pytest.fixture
def dummy_data(tmp_path):
    df = pd.DataFrame({
        "UserId": ["U1", "U2", "U3"],
        "ProductId": ["P1", "P2", "P3"],
        "Score": [5, 4, 3]
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    # Dummy model and svd
    model_path = tmp_path / "model.pkl"
    svd_path = tmp_path / "svd.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(None, f)
    with open(svd_path, "wb") as f:
        pickle.dump(None, f)
    return str(data_path), str(model_path), str(svd_path)

def test_top_n(dummy_data):
    data_path, model_path, svd_path = dummy_data
    rec = Recommender(data_path, model_path, svd_path)
    assert isinstance(rec.top_n("U1"), list)
