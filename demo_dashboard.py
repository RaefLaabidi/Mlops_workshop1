"""
Demo script to showcase the Real-time Training Dashboard.
Run this to see the beautiful training visualization!
"""
from training_dashboard import TrainingDashboard
import time
import numpy as np


def demo_dashboard():
    """Demonstrate the training dashboard features."""
    dashboard = TrainingDashboard()
    
    # 1. Show training header
    params = {
        'Model': 'Random Forest',
        'N Estimators': 200,
        'Max Depth': 10,
        'Random State': 42,
        'Training Samples': 1000
    }
    dashboard.show_header("Random Forest Regressor", params)
    time.sleep(2)
    
    # 2. Show training progress
    print("\n")
    n_estimators = 100
    progress = dashboard.show_training_progress(n_estimators)
    
    with progress:
        task = progress.add_task("Training", total=n_estimators)
        
        for i in range(n_estimators):
            # Simulate training
            time.sleep(0.02)
            
            # Update metrics every 10 steps
            if i % 10 == 0:
                metrics = {
                    'r2': 0.65 + (i / n_estimators) * 0.28 + np.random.uniform(-0.02, 0.02),
                    'rmse': 0.15 - (i / n_estimators) * 0.08 + np.random.uniform(-0.01, 0.01),
                    'mae': 0.12 - (i / n_estimators) * 0.05 + np.random.uniform(-0.005, 0.005)
                }
                dashboard.show_live_metrics(metrics)
            
            progress.update(task, advance=1)
    
    # 3. Show completion summary
    final_metrics = {
        'r2': 0.9234,
        'rmse': 0.0265,
        'mae': 0.0189
    }
    dashboard.show_completion_summary(final_metrics)
    time.sleep(2)
    
    # 4. Show cross-validation
    cv_folds = 5
    progress = dashboard.show_cross_validation_progress(cv_folds)
    
    cv_scores = []
    with progress:
        task = progress.add_task("CV", total=cv_folds)
        
        for fold in range(cv_folds):
            # Simulate CV fold
            time.sleep(1)
            score = 0.88 + np.random.uniform(-0.05, 0.07)
            cv_scores.append(score)
            progress.update(task, advance=1)
    
    dashboard.show_cv_results(cv_scores)
    
    # 5. Show info/warning/error examples
    time.sleep(2)
    dashboard.show_info("Model saved successfully to 'random_forest_model.joblib'")
    time.sleep(1)
    dashboard.show_warning("Test set R² is lower than training R². Possible overfitting.")
    time.sleep(1)
    
    dashboard.console.print("\n[bold green]✅ Demo completed! This is what your training will look like.[/bold green]\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 Real-time Training Dashboard Demo".center(70))
    print("="*70 + "\n")
    
    try:
        demo_dashboard()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted.")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please install required packages:")
        print("   pip install rich")
