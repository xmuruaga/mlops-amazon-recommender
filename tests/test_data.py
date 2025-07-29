import pandas as pd
import pytest
from src.amazon_recommender.data import load_and_clean

@pytest.fixture
def raw_df(tmp_path):
    df = pd.DataFrame({
        "UserId": ["A", "A", "B", None],
        "ProductId": ["X", "X", "Y", "Z"],
        "Score": [5, 5, 4, 3],
        "Time": [1000, 1000, 2000, 3000]
    })
    data_path = tmp_path / "raw.csv"
    df.to_csv(data_path, index=False)
    return str(data_path)

def test_load_and_clean_dedup_and_nulls(raw_df):
    clean_df = load_and_clean(raw_df)
    assert clean_df.shape == (2, 4)
    assert clean_df["UserId"].isnull().sum() == 0
