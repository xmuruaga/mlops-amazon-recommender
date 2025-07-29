def test_sparse_shape():
    from src.amazon_recommender.model import build_sparse_matrix
    from src.amazon_recommender.config import CFG
    import pandas as pd
    # Use config values if needed (example: n_users, n_items)
    df = pd.DataFrame({"UserId": ["u1"], "ProductId": ["i1"], "Score": [5]})
    M, ui, ii = build_sparse_matrix(df)
    assert M.shape == (1, 1)
    # Example: check config value
    assert isinstance(CFG["dataset"]["n_users"], int)
