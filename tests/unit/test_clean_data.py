import pandas as pd
from model_pipeline import clean_data

def test_clean_data_removes_nulls():
    df = pd.DataFrame({
        "department": ["A", None, "C"],
        "quarter": ["Q1", "Q2", "Q3"],
        "day": ["Mon", None, "Wed"],
        "wip": [10, None, 30],
        "actual_productivity": [100, 200, 300],
        "over_time": [0, 5, 10],
        "no_of_workers": [2, 4, 1],
        "idle_time": [0, 0, 0],
        "idle_men": [1, 0, 1],
        "incentive": [5, 10, 15],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    })
    cleaned_df = clean_data(df)
    assert cleaned_df.isnull().sum().sum() == 0  # No nulls left
    assert "date" in cleaned_df.columns
