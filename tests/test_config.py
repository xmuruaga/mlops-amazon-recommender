def test_config_wiring():
    from src.amazon_recommender.config import CFG
    # Check that config loads and n_users is an int
    assert isinstance(CFG["dataset"]["n_users"], int)
    # Example: check default value
    assert CFG["dataset"]["n_users"] == 5000
