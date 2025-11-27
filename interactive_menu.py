"""
Professional Interactive Menu for Garment Productivity Prediction Pipeline.
Integrates with Makefile targets and Python scripts.
"""
import os
import sys
import subprocess
from typing import Callable, Dict, List, Tuple


class InteractiveMenu:
    """Professional CLI menu system with navigation and validation."""

    def __init__(self):
        """Initialize the menu system."""
        self.running = True
        self.menu_stack = []
        self.is_windows = os.name == 'nt'
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if self.is_windows else 'clear')
    
    def print_header(self, title: str):
        """Print a styled header."""
        width = 70
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width + "\n")
    
    def print_menu_options(self, options: List[Tuple[str, str]]):
        """Print numbered menu options."""
        for idx, (key, description) in enumerate(options, 1):
            print(f"  [{idx}] {description}")
        print(f"\n  [0] {'Exit' if not self.menu_stack else 'Back'}")
        print()
    
    def get_user_choice(self, max_option: int) -> int:
        """Get and validate user input."""
        while True:
            try:
                choice = input("Enter your choice: ").strip()
                choice_int = int(choice)
                if 0 <= choice_int <= max_option:
                    return choice_int
                else:
                    print(f"❌ Invalid choice. Please enter 0-{max_option}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                sys.exit(0)
    
    def run_command(self, command: str, description: str):
        """Execute a shell command and display results."""
        self.clear_screen()
        self.print_header(description)
        
        print(f"🔄 Executing: {command}\n")
        print("-" * 70)
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=False
            )
            
            print("-" * 70)
            if result.returncode == 0:
                print(f"\n✅ {description} completed successfully!")
            else:
                print(f"\n⚠️ {description} completed with warnings/errors.")
                
        except Exception as e:
            print(f"\n❌ Error executing command: {e}")
        
        input("\n\nPress Enter to continue...")
    
    def run_python_script(self, script: str, args: str, description: str):
        """Run a Python script with arguments."""
        if self.is_windows:
            python_cmd = "venv\\Scripts\\python.exe"
        else:
            python_cmd = "venv/bin/python3"
        
        command = f"{python_cmd} {script} {args}"
        self.run_command(command, description)
    
    def run_make_target(self, target: str, description: str):
        """Run a Makefile target."""
        command = f"make {target}"
        self.run_command(command, description)
    
    # ========== MAIN MENU ==========
    def main_menu(self):
        """Display and handle main menu."""
        while self.running:
            self.clear_screen()
            self.print_header("🎯 ML Pipeline - Main Menu")
            
            options = [
                ("setup", "🔧 Environment Setup"),
                ("pipeline", "🚀 Pipeline Operations"),
                ("quality", "✨ Code Quality & Testing"),
                ("models", "🤖 Model Management"),
                ("utilities", "🛠️  Utilities & Tools"),
            ]
            
            self.print_menu_options(options)
            choice = self.get_user_choice(len(options))
            
            if choice == 0:
                self.confirm_exit()
            elif choice == 1:
                self.environment_menu()
            elif choice == 2:
                self.pipeline_menu()
            elif choice == 3:
                self.quality_menu()
            elif choice == 4:
                self.model_menu()
            elif choice == 5:
                self.utilities_menu()
    
    # ========== SUB MENUS ==========
    def environment_menu(self):
        """Environment setup and configuration menu."""
        self.menu_stack.append("main")
        
        while True:
            self.clear_screen()
            self.print_header("🔧 Environment Setup")
            
            options = [
                ("setup", "Setup Virtual Environment & Dependencies"),
                ("check", "Check Environment Status"),
                ("update", "Update Dependencies"),
            ]
            
            self.print_menu_options(options)
            choice = self.get_user_choice(len(options))
            
            if choice == 0:
                self.menu_stack.pop()
                break
            elif choice == 1:
                self.run_make_target("setup", "Environment Setup")
            elif choice == 2:
                self.check_environment()
            elif choice == 3:
                self.update_dependencies()
    
    def pipeline_menu(self):
        """Pipeline operations menu."""
        self.menu_stack.append("main")
        
        while True:
            self.clear_screen()
            self.print_header("🚀 Pipeline Operations")
            
            options = [
                ("prepare", "📁 Prepare Data Only"),
                ("train", "🤖 Train Model Only"),
                ("evaluate", "📊 Evaluate Model Only"),
                ("all", "🔄 Run Complete Pipeline"),
                ("custom", "⚙️  Custom Pipeline Run"),
            ]
            
            self.print_menu_options(options)
            choice = self.get_user_choice(len(options))
            
            if choice == 0:
                self.menu_stack.pop()
                break
            elif choice == 1:
                self.run_make_target("data", "Data Preparation")
            elif choice == 2:
                self.run_make_target("train", "Model Training")
            elif choice == 3:
                self.run_python_script("main2.py", "--evaluate", "Model Evaluation")
            elif choice == 4:
                self.run_python_script("main2.py", "--all", "Complete Pipeline")
            elif choice == 5:
                self.custom_pipeline()
    
    def quality_menu(self):
        """Code quality and testing menu."""
        self.menu_stack.append("main")
        
        while True:
            self.clear_screen()
            self.print_header("✨ Code Quality & Testing")
            
            options = [
                ("quality", "🔍 Run All Quality Checks"),
                ("pylint", "📝 Run Pylint"),
                ("flake8", "🎨 Run Flake8"),
                ("bandit", "🔒 Run Bandit Security Check"),
                ("test", "🧪 Run Unit Tests"),
                ("integration", "🔗 Run Integration Tests"),
                ("functional", "⚙️  Run Functional Tests"),
                ("all_tests", "🎯 Run All Tests"),
            ]
            
            self.print_menu_options(options)
            choice = self.get_user_choice(len(options))
            
            if choice == 0:
                self.menu_stack.pop()
                break
            elif choice == 1:
                self.run_make_target("quality", "Code Quality Checks")
            elif choice == 2:
                self.run_make_target("pylint", "Pylint Analysis")
            elif choice == 3:
                self.run_make_target("flake8", "Flake8 Analysis")
            elif choice == 4:
                self.run_make_target("bandit", "Bandit Security Check")
            elif choice == 5:
                self.run_make_target("test", "Unit Tests")
            elif choice == 6:
                self.run_make_target("integration", "Integration Tests")
            elif choice == 7:
                self.run_make_target("functional", "Functional Tests")
            elif choice == 8:
                self.run_all_tests()
    
    def model_menu(self):
        """Model management menu."""
        self.menu_stack.append("main")
        
        while True:
            self.clear_screen()
            self.print_header("🤖 Model Management")
            
            options = [
                ("train", "🎓 Train New Model"),
                ("evaluate", "📊 Evaluate Existing Model"),
                ("compare", "⚖️  Compare Models (Coming Soon)"),
                ("export", "💾 Export Model"),
                ("info", "ℹ️  View Model Information"),
            ]
            
            self.print_menu_options(options)
            choice = self.get_user_choice(len(options))
            
            if choice == 0:
                self.menu_stack.pop()
                break
            elif choice == 1:
                self.run_make_target("train", "Model Training")
            elif choice == 2:
                self.run_python_script("main2.py", "--evaluate", "Model Evaluation")
            elif choice == 3:
                print("\n⚠️  Feature coming soon!")
                input("Press Enter to continue...")
            elif choice == 4:
                self.export_model()
            elif choice == 5:
                self.view_model_info()
    
    def utilities_menu(self):
        """Utilities and tools menu."""
        self.menu_stack.append("main")
        
        while True:
            self.clear_screen()
            self.print_header("🛠️  Utilities & Tools")
            
            options = [
                ("notebook", "📓 Launch Jupyter Notebook"),
                ("clean", "🧹 Clean Generated Files"),
                ("logs", "📋 View Logs"),
                ("help", "❓ Help & Documentation"),
            ]
            
            self.print_menu_options(options)
            choice = self.get_user_choice(len(options))
            
            if choice == 0:
                self.menu_stack.pop()
                break
            elif choice == 1:
                self.run_make_target("notebook", "Jupyter Notebook")
            elif choice == 2:
                self.clean_files()
            elif choice == 3:
                self.view_logs()
            elif choice == 4:
                self.show_help()
    
    # ========== UTILITY FUNCTIONS ==========
    def check_environment(self):
        """Check environment status."""
        self.clear_screen()
        self.print_header("Environment Status Check")
        
        print("Checking Python environment...\n")
        
        if self.is_windows:
            python_cmd = "venv\\Scripts\\python.exe"
        else:
            python_cmd = "venv/bin/python3"
        
        checks = [
            (f"{python_cmd} --version", "Python Version"),
            (f"{python_cmd} -m pip --version", "Pip Version"),
            (f"{python_cmd} -m pip list", "Installed Packages"),
        ]
        
        for cmd, desc in checks:
            print(f"🔍 {desc}:")
            print("-" * 70)
            subprocess.run(cmd, shell=True)
            print()
        
        input("\nPress Enter to continue...")
    
    def update_dependencies(self):
        """Update project dependencies."""
        self.clear_screen()
        self.print_header("Update Dependencies")
        
        if self.is_windows:
            python_cmd = "venv\\Scripts\\python.exe"
        else:
            python_cmd = "venv/bin/python3"
        
        commands = [
            (f"{python_cmd} -m pip install --upgrade pip", "Upgrading pip"),
            (f"{python_cmd} -m pip install -r requirements.txt --upgrade", "Updating packages"),
        ]
        
        for cmd, desc in commands:
            print(f"\n🔄 {desc}...")
            print("-" * 70)
            subprocess.run(cmd, shell=True)
            print()
        
        print("✅ Dependencies updated!")
        input("\nPress Enter to continue...")
    
    def custom_pipeline(self):
        """Run custom pipeline with user-selected stages."""
        self.clear_screen()
        self.print_header("⚙️  Custom Pipeline Configuration")
        
        print("Select pipeline stages to run:\n")
        print("  [1] Prepare Data")
        print("  [2] Train Model")
        print("  [3] Evaluate Model")
        print("\nEnter stage numbers separated by commas (e.g., 1,2,3): ")
        
        try:
            stages_input = input().strip()
            stages = [int(s.strip()) for s in stages_input.split(',')]
            
            args = []
            if 1 in stages:
                args.append("--prepare")
            if 2 in stages:
                args.append("--train")
            if 3 in stages:
                args.append("--evaluate")
            
            if args:
                args_str = " ".join(args)
                self.run_python_script("main2.py", args_str, "Custom Pipeline")
            else:
                print("\n⚠️  No valid stages selected.")
                input("Press Enter to continue...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")
    
    def run_all_tests(self):
        """Run all test suites sequentially."""
        self.clear_screen()
        self.print_header("Running All Tests")
        
        test_types = [
            ("test", "Unit Tests"),
            ("integration", "Integration Tests"),
            ("functional", "Functional Tests"),
        ]
        
        for target, name in test_types:
            print(f"\n🧪 Running {name}...")
            print("-" * 70)
            subprocess.run(f"make {target}", shell=True)
            print()
        
        print("✅ All tests completed!")
        input("\nPress Enter to continue...")
    
    def export_model(self):
        """Export trained model."""
        self.clear_screen()
        self.print_header("💾 Export Model")
        
        print("Checking for trained models...\n")
        
        if os.path.exists("random_forest_model.joblib"):
            print("✅ Found: random_forest_model.joblib")
            print("\nModel is already saved and ready to use!")
        else:
            print("❌ No trained model found.")
            print("Please train a model first using the Pipeline menu.")
        
        input("\nPress Enter to continue...")
    
    def view_model_info(self):
        """Display information about trained models."""
        self.clear_screen()
        self.print_header("ℹ️  Model Information")
        
        print("Checking for model files...\n")
        
        model_files = [
            "random_forest_model.joblib",
            "prepared_data.joblib"
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                size = os.path.getsize(model_file)
                print(f"✅ {model_file}")
                print(f"   Size: {size:,} bytes")
                print()
            else:
                print(f"❌ {model_file} - Not found")
                print()
        
        input("Press Enter to continue...")
    
    def clean_files(self):
        """Clean generated files and cache."""
        self.clear_screen()
        self.print_header("🧹 Clean Generated Files")
        
        print("This will remove:\n")
        print("  • Compiled Python files (__pycache__, *.pyc)")
        print("  • Model files (*.joblib)")
        print("  • Log files")
        print()
        
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            patterns = [
                ("__pycache__", "Python cache directories"),
                ("*.pyc", "Compiled Python files"),
                ("*.joblib", "Model files"),
                (".pytest_cache", "Pytest cache"),
            ]
            
            print("\n🔄 Cleaning...\n")
            
            for pattern, desc in patterns:
                print(f"  Removing {desc}...")
                if self.is_windows:
                    if pattern.startswith("*"):
                        subprocess.run(f"del /S /Q {pattern}", shell=True, 
                                     capture_output=True)
                    else:
                        subprocess.run(f"rmdir /S /Q {pattern}", shell=True, 
                                     capture_output=True)
                else:
                    subprocess.run(f"find . -type d -name '{pattern}' -exec rm -rf {{}} +", 
                                 shell=True, capture_output=True)
            
            print("\n✅ Cleanup completed!")
        else:
            print("\n❌ Cleanup cancelled.")
        
        input("\nPress Enter to continue...")
    
    def view_logs(self):
        """View system logs."""
        self.clear_screen()
        self.print_header("📋 View Logs")
        
        print("⚠️  Log viewing feature coming soon!")
        print("\nFor now, check the console output when running pipeline operations.")
        
        input("\nPress Enter to continue...")
    
    def show_help(self):
        """Display help information."""
        self.clear_screen()
        self.print_header("❓ Help & Documentation")
        
        help_text = """
🎯 PIPELINE OVERVIEW:
   This is a Machine Learning pipeline for Garment Productivity Prediction.

📚 MAIN FEATURES:
   • Data preparation and cleaning
   • Feature engineering
   • Model training (Random Forest)
   • Model evaluation and cross-validation
   • Code quality checks
   • Comprehensive testing suite

🚀 QUICK START:
   1. Setup environment (Environment Menu → Setup)
   2. Run complete pipeline (Pipeline Menu → Run Complete Pipeline)
   3. Evaluate results (Pipeline Menu → Evaluate Model)

📂 GENERATED FILES:
   • prepared_data.joblib - Preprocessed training/test data
   • random_forest_model.joblib - Trained model

🔧 MAKEFILE COMMANDS:
   You can also run commands directly:
   • make setup     - Setup environment
   • make train     - Train model
   • make test      - Run tests
   • make quality   - Run quality checks
   • make all       - Run complete pipeline

📖 DOCUMENTATION:
   Check the source files for detailed function documentation.
        """
        
        print(help_text)
        input("\nPress Enter to continue...")
    
    def confirm_exit(self):
        """Confirm before exiting."""
        print("\n👋 Are you sure you want to exit? (yes/no): ", end='')
        confirm = input().strip().lower()
        if confirm in ['yes', 'y']:
            self.clear_screen()
            print("\n" + "=" * 70)
            print("Thank you for using the ML Pipeline Interactive Menu!".center(70))
            print("=" * 70 + "\n")
            self.running = False
            sys.exit(0)


def main():
    """Entry point for the interactive menu."""
    try:
        menu = InteractiveMenu()
        menu.main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
