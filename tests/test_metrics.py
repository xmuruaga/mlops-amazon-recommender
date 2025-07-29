import numpy as np
from sklearn.metrics import mean_squared_error

def test_rmse_and_metrics():
    actual = np.array([5, 4, 3])
    pred   = np.array([5, 4, 2])
    rmse = np.sqrt(mean_squared_error(actual, pred))
    assert np.isclose(rmse, 0.577, atol=0.01)
    recommended = ["A", "B", "C"]
    relevant    = ["A", "B", "C"]
    hits = len(set(recommended) & set(relevant))
    precision = hits / len(recommended)
    assert precision == 1.0
    def ndcg(recommended, relevant):
        dcg = sum([1/np.log2(i+2) for i, item in enumerate(recommended) if item in relevant])
        ideal = sum([1/np.log2(i+2) for i in range(min(len(recommended), len(relevant)))])
        return dcg / ideal if ideal else 0
    score = ndcg(["A", "B", "C"], ["A", "B", "C"])
    assert 0 <= score <= 1
