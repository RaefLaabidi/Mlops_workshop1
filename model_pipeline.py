"""
Machine Learning Pipeline for Garment Productivity Prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import time

# Import training dashboard
try:
    from training_dashboard import TrainingDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("⚠️  Training dashboard not available. Install 'rich' for better experience.")


def load_data(file_path):
    """Load the garment productivity dataset."""
    data_frame = pd.read_csv(file_path)
    print(f"✅ Dataset loaded successfully. Shape: {data_frame.shape}")
    return data_frame


def explore_data(data_frame):
    """Explore dataset structure and statistics."""
    print("*** Dataset Info ***:")
    print(data_frame.info())

    print("\n*** Missing Values per Column ***:")
    print(data_frame.isnull().sum())

    print("\n*** Summary Statistics ***:")
    print(data_frame.describe())

    return data_frame


def clean_data(data_frame):
    """Clean and preprocess the data."""
    print("🔄 Cleaning data...")

    for col in ['department', 'quarter', 'day']:
        data_frame[col] = data_frame[col].astype(str).str.strip()

    data_frame['quarter'] = data_frame['quarter'].replace({'Quarter5': 'Quarter4'})
    data_frame['date'] = pd.to_datetime(data_frame['date'], errors='coerce')

    numeric_cols = data_frame.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('wip')

    for col in numeric_cols:
        data_frame[col] = data_frame[col].fillna(data_frame[col].mean())

    cat_cols = ['day', 'department', 'quarter']
    for col in cat_cols:
        data_frame[col] = data_frame[col].replace('nan', np.nan)
        data_frame[col] = data_frame[col].fillna(data_frame[col].mode()[0])

    data_frame['wip'] = data_frame['wip'].fillna(data_frame['wip'].median())
    data_frame = data_frame.dropna(subset=['actual_productivity'])

    data_frame = data_frame[
        (data_frame['over_time'] >= -6840.0) &
        (data_frame['over_time'] <= 15240.0)
    ]

    print(f"✅ Data cleaned. New shape: {data_frame.shape}")
    return data_frame


def engineer_features(data_frame):
    """Create new features for better model performance."""
    print("🔄 Engineering features...")

    data_frame['wip_per_worker'] = data_frame['wip'] / data_frame['no_of_workers']
    data_frame['over_time_per_worker'] = (
        data_frame['over_time'] / data_frame['no_of_workers']
    )
    data_frame['idle_time_per_worker'] = (
        data_frame['idle_time'] / data_frame['idle_men'].replace(0, 1)
    )

    data_frame['wip_x_workers'] = data_frame['wip'] * data_frame['no_of_workers']
    data_frame['incentive_per_worker'] = (
        data_frame['incentive'] / (data_frame['no_of_workers'] + 1e-6)
    )

    print("✅ Feature engineering completed")
    return data_frame


def prepare_data(data_frame, test_size=0.2, random_state=42):
    """Prepare data for training (train/test split and preprocessing)."""
    print("🔄 Preparing data for training...")

    feature_cols = data_frame.drop(
        columns=['actual_productivity', 'date']
    ).columns.tolist()

    categorical_cols = ['day', 'department', 'quarter']
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'),
             categorical_cols)
        ]
    )

    features = data_frame[feature_cols]
    target = data_frame['actual_productivity']

    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )

    train_shape = features_train.shape
    test_shape = features_test.shape
    msg = f"✅ Data prepared. Train shape: {train_shape}, Test shape: {test_shape}"
    print(msg)
    return features_train, features_test, target_train, target_test, preprocessor


def train_model(features_train, target_train, preprocessor,
                model_name='random_forest', **kwargs):
    """Train the Random Forest model with real-time dashboard."""
    
    # Initialize dashboard if available
    dashboard = TrainingDashboard() if DASHBOARD_AVAILABLE else None
    use_dashboard = kwargs.get('use_dashboard', True) and DASHBOARD_AVAILABLE
    
    n_estimators = kwargs.get('n_estimators', 200)
    random_state = kwargs.get('random_state', 42)
    
    if use_dashboard and dashboard:
        # Show header with configuration
        params = {
            'Model': model_name,
            'N Estimators': n_estimators,
            'Random State': random_state,
            'Features': features_train.shape[1],
            'Training Samples': features_train.shape[0]
        }
        dashboard.show_header(model_name.replace('_', ' ').title(), params)
    else:
        print(f"🔄 Training {model_name} model...")

    model_instance = None
    if model_name == 'random_forest':
        model_instance = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=0,
            warm_start=False
        )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model_instance)
    ])

    # Train with progress tracking
    start_time = time.time()
    
    if use_dashboard and dashboard:
        progress = dashboard.show_training_progress(n_estimators)
        with progress:
            task = progress.add_task("Training", total=100)
            
            # Fit the model
            pipeline.fit(features_train, target_train)
            progress.update(task, completed=100)
        
        # Calculate training metrics
        train_predictions = pipeline.predict(features_train)
        train_metrics = {
            'r2': r2_score(target_train, train_predictions),
            'rmse': np.sqrt(mean_squared_error(target_train, train_predictions)),
            'mae': mean_absolute_error(target_train, train_predictions)
        }
        
        dashboard.show_completion_summary(train_metrics)
    else:
        pipeline.fit(features_train, target_train)
        training_time = time.time() - start_time
        print(f"✅ {model_name} model trained successfully in {training_time:.2f}s")
    
    return pipeline


def evaluate_model(model, features_test, target_test, model_name="Random Forest", use_dashboard=True):
    """Evaluate model performance with dashboard display."""
    
    dashboard = TrainingDashboard() if (DASHBOARD_AVAILABLE and use_dashboard) else None
    
    if dashboard:
        dashboard.console.print(f"\n[bold cyan]📊 Evaluating {model_name}...[/bold cyan]\n")
    else:
        print(f"📊 Evaluating {model_name}...")

    predictions = model.predict(features_test)

    rmse = np.sqrt(mean_squared_error(target_test, predictions))
    mae = mean_absolute_error(target_test, predictions)
    r2 = r2_score(target_test, predictions)

    if dashboard:
        metrics = {'r2': r2, 'rmse': rmse, 'mae': mae}
        dashboard.show_live_metrics(metrics)
    else:
        print(f"✅ {model_name} Evaluation Results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }


def cross_validate_model(model, features, target, cv=5, use_dashboard=True):
    """Perform cross-validation with real-time dashboard."""
    
    # Initialize dashboard if available
    dashboard = TrainingDashboard() if (DASHBOARD_AVAILABLE and use_dashboard) else None
    
    if dashboard:
        progress = dashboard.show_cross_validation_progress(cv)
        scores = []
        
        with progress:
            task = progress.add_task("Cross-Validation", total=cv)
            
            # Manual cross-validation to show progress
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(features)):
                X_train_fold = features.iloc[train_idx]
                X_val_fold = features.iloc[val_idx]
                y_train_fold = target.iloc[train_idx]
                y_val_fold = target.iloc[val_idx]
                
                # Fit and score
                model.fit(X_train_fold, y_train_fold)
                score = model.score(X_val_fold, y_val_fold)
                scores.append(score)
                
                progress.update(task, advance=1)
        
        scores = np.array(scores)
        dashboard.show_cv_results(scores)
    else:
        print("🔄 Performing cross-validation...")
        scores = cross_val_score(
            model, features, target, cv=cv, scoring='r2'
        )
        print(f"✅ Cross-Validation R² scores: {scores}")
        print(f"   Mean R²: {np.mean(scores):.4f}")
        print(f"   Std of R²: {np.std(scores):.4f}")

    return scores


def save_model(model, filepath):
    """Save trained model using joblib."""
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")


def load_model(filepath):
    """Load saved model."""
    model = joblib.load(filepath)
    print(f"✅ Model loaded from {filepath}")
    return model


def plot_predictions(target_true, target_pred, model_name="Random Forest"):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(target_true, target_pred, alpha=0.5)
    plt.plot([target_true.min(), target_true.max()],
             [target_true.min(), target_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Productivity')
    plt.ylabel('Predicted Productivity')
    plt.title(f'{model_name}: Actual vs Predicted Productivity')
    plt.tight_layout()
    plt.show()