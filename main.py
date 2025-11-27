"""
Main script for Garment Productivity Prediction Pipeline.
"""
from model_pipeline import (
    load_data, explore_data, clean_data, engineer_features,
    prepare_data, train_model, evaluate_model, cross_validate_model,
    save_model, load_model
)


def main():
    """Execute the complete garment productivity prediction pipeline."""
    print("🚀 Starting Garment Productivity Prediction Pipeline...")

    data_file = "dataProductivity-Prediction-of-Garment-Employeese.csv"
    data_frame = load_data(data_file)
    explore_data(data_frame)
    data_frame_clean = clean_data(data_frame)
    data_frame_featured = engineer_features(data_frame_clean)

    (features_train, features_test, target_train,
     target_test, preprocessor) = prepare_data(data_frame_featured)

    model = train_model(
        features_train, target_train, preprocessor,
        model_name='random_forest',
        n_estimators=200,
        random_state=42
    )

    evaluate_model(model, features_test, target_test)

    features_full = data_frame_featured.drop(
        columns=['actual_productivity', 'date']
    )
    target_full = data_frame_featured['actual_productivity']
    cross_validate_model(model, features_full, target_full, cv=5)

    save_model(model, 'random_forest_model.joblib')
    loaded_model = load_model('random_forest_model.joblib')

    print("\n🧪 Testing loaded model...")
    evaluate_model(loaded_model, features_test, target_test, "Loaded Random Forest")

    print("\n🎯 Pipeline completed successfully!")


if __name__ == "__main__":
    main()