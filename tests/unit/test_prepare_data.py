import pandas as pd
from model_pipeline import prepare_data

def test_prepare_data_splits_correctly():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "department": ["A", "B", "A", "B", "C"],
        "quarter": ["Q1", "Q2", "Q1", "Q2", "Q3"],
        "actual_productivity": [10, 20, 30, 40, 50],
        "date": pd.to_datetime(["2025-01-01","2025-01-02","2025-01-03","2025-01-04","2025-01-05"])
    })
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, test_size=0.4, random_state=42)
    assert len(X_train) == 3
    assert len(X_test) == 2
    assert y_train.sum() + y_test.sum() == df['actual_productivity'].sum()
