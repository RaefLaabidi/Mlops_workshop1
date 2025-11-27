"""
Real-time Training Dashboard with Rich Progress Display.
"""
import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn
)
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box
import numpy as np


class TrainingDashboard:
    """Professional real-time training dashboard with live metrics."""

    def __init__(self):
        """Initialize the dashboard."""
        self.console = Console()
        self.metrics_history = {
            'r2': [],
            'rmse': [],
            'mae': [],
            'time': []
        }
        self.start_time = None
        self.current_estimator = 0
        self.total_estimators = 0

    def show_header(self, model_name: str, params: Dict[str, Any]):
        """Display training header with model information."""
        self.console.clear()
        
        # Create header panel
        header_text = f"[bold cyan]🤖 Training {model_name} Model[/bold cyan]\n\n"
        header_text += "[yellow]Configuration:[/yellow]\n"
        for key, value in params.items():
            header_text += f"  • {key}: [green]{value}[/green]\n"
        
        panel = Panel(
            header_text,
            border_style="cyan",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()

    def create_progress_bar(self, total: int, description: str = "Training"):
        """Create a rich progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )

    def create_metrics_table(self) -> Table:
        """Create a beautiful metrics table."""
        table = Table(
            title="📊 Training Metrics",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan"
        )
        
        table.add_column("Metric", style="cyan", justify="left")
        table.add_column("Current", style="green", justify="right")
        table.add_column("Best", style="yellow", justify="right")
        table.add_column("Trend", justify="center")
        
        return table

    def update_metrics_table(self, metrics: Dict[str, float]) -> Table:
        """Update the metrics table with current values."""
        table = self.create_metrics_table()
        
        # Store metrics
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Calculate trends
        def get_trend(history):
            if len(history) < 2:
                return "➡️"
            if history[-1] > history[-2]:
                return "📈"
            elif history[-1] < history[-2]:
                return "📉"
            return "➡️"
        
        # Add metrics to table
        if 'r2' in metrics:
            best_r2 = max(self.metrics_history['r2']) if self.metrics_history['r2'] else 0
            table.add_row(
                "R² Score",
                f"{metrics['r2']:.4f}",
                f"{best_r2:.4f}",
                get_trend(self.metrics_history['r2'])
            )
        
        if 'rmse' in metrics:
            best_rmse = min(self.metrics_history['rmse']) if self.metrics_history['rmse'] else float('inf')
            table.add_row(
                "RMSE",
                f"{metrics['rmse']:.4f}",
                f"{best_rmse:.4f}",
                get_trend([1/x for x in self.metrics_history['rmse']])
            )
        
        if 'mae' in metrics:
            best_mae = min(self.metrics_history['mae']) if self.metrics_history['mae'] else float('inf')
            table.add_row(
                "MAE",
                f"{metrics['mae']:.4f}",
                f"{best_mae:.4f}",
                get_trend([1/x for x in self.metrics_history['mae']])
            )
        
        # Add training time
        if self.start_time:
            elapsed = time.time() - self.start_time
            table.add_row(
                "Training Time",
                f"{elapsed:.1f}s",
                "-",
                "⏱️"
            )
        
        return table

    def show_training_progress(self, n_estimators: int):
        """Show training progress with live updates."""
        self.total_estimators = n_estimators
        self.start_time = time.time()
        
        return self.create_progress_bar(
            n_estimators,
            "Training Random Forest"
        )

    def show_completion_summary(self, final_metrics: Dict[str, float]):
        """Display beautiful completion summary."""
        self.console.print()
        
        # Create summary panel
        summary_text = "[bold green]✅ Training Completed Successfully![/bold green]\n\n"
        summary_text += "[bold yellow]📊 Final Results:[/bold yellow]\n"
        
        if 'r2' in final_metrics:
            summary_text += f"  • R² Score: [bold green]{final_metrics['r2']:.4f}[/bold green]\n"
        if 'rmse' in final_metrics:
            summary_text += f"  • RMSE: [bold cyan]{final_metrics['rmse']:.4f}[/bold cyan]\n"
        if 'mae' in final_metrics:
            summary_text += f"  • MAE: [bold cyan]{final_metrics['mae']:.4f}[/bold cyan]\n"
        
        if self.start_time:
            total_time = time.time() - self.start_time
            summary_text += f"\n[bold yellow]⏱️  Total Time:[/bold yellow] [green]{total_time:.2f}s[/green]"
        
        panel = Panel(
            summary_text,
            border_style="green",
            box=box.DOUBLE,
            padding=(1, 2),
            title="[bold green]Training Summary[/bold green]"
        )
        
        self.console.print(panel)
        self.console.print()

    def show_live_metrics(self, metrics: Dict[str, float]):
        """Display live updating metrics table."""
        table = self.update_metrics_table(metrics)
        self.console.print(table)

    def show_cross_validation_progress(self, cv_folds: int):
        """Show cross-validation progress."""
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]🔄 Performing {cv_folds}-Fold Cross-Validation[/bold cyan]",
            border_style="cyan"
        ))
        
        return self.create_progress_bar(
            cv_folds,
            "Cross-Validation"
        )

    def show_cv_results(self, scores: list):
        """Display cross-validation results."""
        self.console.print()
        
        table = Table(
            title="📊 Cross-Validation Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        table.add_column("Fold", style="cyan", justify="center")
        table.add_column("R² Score", style="green", justify="right")
        table.add_column("Performance", justify="center")
        
        for i, score in enumerate(scores, 1):
            # Determine performance indicator
            if score >= 0.9:
                perf = "🌟 Excellent"
            elif score >= 0.8:
                perf = "✅ Good"
            elif score >= 0.7:
                perf = "⚠️  Fair"
            else:
                perf = "❌ Poor"
            
            table.add_row(f"Fold {i}", f"{score:.4f}", perf)
        
        # Add summary row
        table.add_section()
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        table.add_row(
            "[bold]Mean ± Std[/bold]",
            f"[bold]{mean_score:.4f} ± {std_score:.4f}[/bold]",
            "[bold]Average[/bold]"
        )
        
        self.console.print(table)
        self.console.print()

    def show_data_preparation_progress(self):
        """Show data preparation progress."""
        steps = [
            ("Loading data", 1.0),
            ("Exploring dataset", 0.5),
            ("Cleaning data", 2.0),
            ("Engineering features", 1.5),
            ("Splitting train/test", 0.5),
            ("Preprocessing", 1.0)
        ]
        
        self.console.print(Panel(
            "[bold cyan]📁 Data Preparation Pipeline[/bold cyan]",
            border_style="cyan"
        ))
        self.console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            for step_name, duration in steps:
                task = progress.add_task(f"[cyan]{step_name}...", total=100)
                
                # Simulate progress
                for i in range(100):
                    progress.update(task, advance=1)
                    time.sleep(duration / 100)
                
                progress.update(task, completed=100)
                progress.stop_task(task)

    def show_error(self, error_message: str):
        """Display error message."""
        self.console.print()
        self.console.print(Panel(
            f"[bold red]❌ Error:[/bold red]\n\n{error_message}",
            border_style="red",
            box=box.DOUBLE
        ))
        self.console.print()

    def show_warning(self, warning_message: str):
        """Display warning message."""
        self.console.print(Panel(
            f"[bold yellow]⚠️  Warning:[/bold yellow] {warning_message}",
            border_style="yellow"
        ))

    def show_info(self, info_message: str):
        """Display info message."""
        self.console.print(Panel(
            f"[bold cyan]ℹ️  Info:[/bold cyan] {info_message}",
            border_style="cyan"
        ))


# Convenience function for simple progress bars
def create_simple_progress(description: str = "Processing"):
    """Create a simple progress bar for general use."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
