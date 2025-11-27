import pandas as pd
from model_pipeline import engineer_features

def test_engineer_features_creates_new_columns():
    df = pd.DataFrame({
        "wip": [10, 20],
        "no_of_workers": [2, 4],
        "over_time": [5, 8],
        "idle_time": [1, 2],
        "idle_men": [1, 2],
        "incentive": [10, 20],
        "department": ["A", "B"],
        "quarter": ["Q1", "Q2"],
        "day": ["Mon", "Tue"],
        "actual_productivity": [100, 200],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"])
    })
    featured_df = engineer_features(df)
    expected_cols = [
        'wip_per_worker', 'over_time_per_worker', 'idle_time_per_worker',
        'wip_x_workers', 'incentive_per_worker'
    ]
    for col in expected_cols:
        assert col in featured_df.columns
