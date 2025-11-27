import os
import joblib
from model_pipeline import load_model

def test_full_pipeline(tmp_path):
    csv_file = tmp_path / "test_data.csv"
    csv_file.write_text(
        "date,department,quarter,day,wip,actual_productivity,over_time,no_of_workers,idle_time,idle_men,incentive\n"
        "2025-01-01,A,Q1,Mon,10,100,5,2,1,1,10\n"
        "2025-01-02,B,Q1,Tue,20,200,8,4,2,2,20\n"
    )
    os.system(f"python3 main2.py --all --file {csv_file}")
    assert os.path.exists("random_forest_model.joblib")
    model = load_model("random_forest_model.joblib")
    assert model is not None
