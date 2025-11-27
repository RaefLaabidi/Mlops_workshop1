"""
Main script with command-line arguments for Garment Productivity Prediction Pipeline.
"""
import argparse
import joblib
import pandas as pd
import pytest  # For running unit tests
from model_pipeline import (
    load_data, explore_data, clean_data, engineer_features,
    prepare_data, train_model, evaluate_model, cross_validate_model,
    save_model, load_model
)


def main():
    """Execute pipeline with command-line options."""
    parser = argparse.ArgumentParser(
        description='Garment Productivity Prediction Pipeline'
    )
    parser.add_argument('--prepare', action='store_true', help='Only prepare data')
    parser.add_argument('--train', action='store_true', help='Only train model')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate model')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--test', action='store_true', help='Run unit tests')  # <-- New

    args = parser.parse_args()

    # If no arguments, run all
    if not any([args.prepare, args.train, args.evaluate, args.all, args.test]):
        args.all = True

    # Run unit tests first if --test is passed
    if args.test:
        print("\n🧪 Running unit tests...")
        # Run pytest in current folder and collect results
        exit_code = pytest.main(["tests/"])  # Assuming your tests are in 'tests/' folder
        if exit_code == 0:
            print("✅ All unit tests passed!")
        else:
            print("❌ Some tests failed!")
        return  # Stop here if only testing

    print("🚀 Starting Garment Productivity Prediction Pipeline...")

    if args.prepare or args.all:
        print("\n📁 STEP 1: Preparing Data...")
        data_file = "dataProductivity-Prediction-of-Garment-Employeese.csv"
        data_frame = load_data(data_file)
        explore_data(data_frame)
        data_frame_clean = clean_data(data_frame)
        data_frame_featured = engineer_features(data_frame_clean)

        (features_train, features_test, target_train,
         target_test, preprocessor) = prepare_data(data_frame_featured)

        joblib.dump({
            'X_train': features_train,
            'X_test': features_test,
            'y_train': target_train,
            'y_test': target_test,
            'preprocessor': preprocessor
        }, 'prepared_data.joblib')
        print("✅ Data preparation completed and saved!")

    if args.train or args.all:
        print("\n🤖 STEP 2: Training Model...")
        try:
            data = joblib.load('prepared_data.joblib')
            features_train = data['X_train']
            target_train = data['y_train']
            preprocessor = data['preprocessor']
        except FileNotFoundError:
            print("❌ No prepared data found. Please run --prepare first.")
            return

        model = train_model(
            features_train, target_train, preprocessor,
            model_name='random_forest',
            n_estimators=200,
            random_state=42
        )

        save_model(model, 'random_forest_model.joblib')
        print("✅ Model training completed!")

    if args.evaluate or args.all:
        print("\n📊 STEP 3: Evaluating Model...")
        try:
            model = load_model('random_forest_model.joblib')
            data = joblib.load('prepared_data.joblib')
            features_test = data['X_test']
            target_test = data['y_test']
        except FileNotFoundError:
            print("❌ Model or data not found. Please run --train first.")
            return

        evaluate_model(model, features_test, target_test)

        features_full = pd.concat([data['X_train'], data['X_test']])
        target_full = pd.concat([data['y_train'], data['y_test']])
        cross_validate_model(model, features_full, target_full, cv=5)

        print("✅ Model evaluation completed!")

    print("\n🎯 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
