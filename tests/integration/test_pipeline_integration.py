from model_pipeline import clean_data, engineer_features, prepare_data
import pandas as pd

def test_pipeline_integration():
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "department": ["A", "B"],
        "quarter": ["Q1", "Q1"],
        "day": ["Mon", "Tue"],
        "wip": [10, 20],
        "actual_productivity": [100, 200],
        "over_time": [5, 8],
        "no_of_workers": [2, 4],
        "idle_time": [1, 2],
        "idle_men": [1, 2],
        "incentive": [10, 20]
    })
    cleaned = clean_data(df)
    featured = engineer_features(cleaned)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(featured, test_size=0.5, random_state=42)
    assert "wip_per_worker" in featured.columns
    assert X_train.shape[0] + X_test.shape[0] == cleaned.shape[0]
