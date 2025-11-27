# VARIABLE DECLARATION
PYTHON = python3
ENV_NAME = venv
PYTHON_BIN = $(ENV_NAME)/bin/python3
REQUIREMENTS = requirements.txt
DATA_FILE = dataProductivity-Prediction-of-Garment-Employeese.csv
PYTHON_FILES = model_pipeline.py main.py main2.py

# ====================================
# 1. ENVIRONMENT CONFIGURATION
# ====================================
.PHONY: setup
setup:
	@echo "🔧 Creating the virtual environment and installing dependencies..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@$(PYTHON_BIN) -m pip install --upgrade pip
	@$(PYTHON_BIN) -m pip install -r $(REQUIREMENTS)

# ====================================
# 2. CODE QUALITY CHECKS
# ====================================
.PHONY: quality
quality: pylint flake8 bandit

.PHONY: pylint
pylint:
	@$(PYTHON_BIN) -m pylint $(PYTHON_FILES) --exit-zero --ignore=R0801,C0301,C0304,W0612,W0613

.PHONY: flake8
flake8:
	@$(PYTHON_BIN) -m flake8 $(PYTHON_FILES) --max-line-length=79 --exit-zero

.PHONY: bandit
bandit:
	@$(PYTHON_BIN) -m bandit -r $(PYTHON_FILES) -f txt

# ====================================
# 3. DATA PREPARATION
# ====================================
.PHONY: data
data:
	@echo "📁 Data preparation..."
	@$(PYTHON_BIN) main2.py --prepare
	@echo "✅ Data preparation complete!"

# ====================================
# 4. MODEL TRAINING
# ====================================
.PHONY: train
train: data
	@echo "🤖 Model training..."
	@$(PYTHON_BIN) main2.py --train
	@echo "✅ Model training complete!"

# ====================================
# 5. UNIT TESTS
# ====================================
.PHONY: test
test:
	@echo "🧪 Running unit tests..."
	@$(PYTHON_BIN) -m pytest tests/unit -q --tb=short
	@echo "✅ Unit tests completed!"

# ====================================
# 6. INTEGRATION TESTS
# ====================================
.PHONY: integration
integration:
	@echo "🧪 Running integration tests..."
	@$(PYTHON_BIN) -m pytest tests/integration -q --tb=short
	@echo "✅ Integration tests completed!"

# ====================================
# 7. FUNCTIONAL TESTS
# ====================================
.PHONY: functional
functional:
	@echo "🧪 Running functional tests..."
	@$(PYTHON_BIN) -m pytest tests/functional -q --tb=short
	@echo "✅ Functional tests completed!"

# ====================================
# 8. JUPYTER NOTEBOOK
# ====================================
.PHONY: notebook
notebook:
	@echo "📓 Starting Jupyter Notebook..."
	@$(PYTHON_BIN) -m jupyter notebook

# ====================================
# 8.5. DASHBOARD DEMO
# ====================================
.PHONY: demo
demo:
	@echo "🎬 Running Dashboard Demo..."
	@$(PYTHON_BIN) demo_dashboard.py
	@echo "✅ Demo completed!"

.PHONY: install-dashboard
install-dashboard:
	@echo "📦 Installing dashboard dependencies..."
	@$(PYTHON_BIN) -m pip install rich tqdm
	@echo "✅ Dashboard dependencies installed!"

# ====================================
# 9. INTERACTIVE MENU
# ====================================
.PHONY: menu
menu:
	@echo "🎯 Launching Interactive Menu..."
	@$(PYTHON_BIN) interactive_menu.py

# ====================================
# 10. CLEAN TARGET
# ====================================
.PHONY: clean
clean:
	@echo "🧹 Cleaning generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -f *.joblib 2>/dev/null || true
	@echo "✅ Cleanup completed!"

# ====================================
# 11. PREDICT TARGET
# ====================================
.PHONY: predict
predict:
	@echo "🔮 Running predictions..."
	@$(PYTHON_BIN) main2.py --evaluate
	@echo "✅ Predictions completed!"

# ====================================
# 12. HELP TARGET
# ====================================
.PHONY: help
help:
	@echo "📚 Available Make Targets:"
	@echo ""
	@echo "  make setup        - Setup virtual environment and dependencies"
	@echo "  make menu         - Launch interactive menu (RECOMMENDED)"
	@echo "  make demo         - See real-time training dashboard demo"
	@echo "  make quality      - Run all code quality checks"
	@echo "  make data         - Prepare data for training"
	@echo "  make train        - Train the model (with dashboard!)"
	@echo "  make test         - Run unit tests"
	@echo "  make integration  - Run integration tests"
	@echo "  make functional   - Run functional tests"
	@echo "  make predict      - Run model predictions"
	@echo "  make notebook     - Start Jupyter Notebook"
	@echo "  make clean        - Clean generated files"
	@echo "  make all          - Run complete pipeline"
	@echo "  make help         - Show this help message"
	@echo ""

# ====================================
# 13. ALL TARGET
# ====================================
.PHONY: all
all: quality data train test integration functional
	@echo "🎯 All steps completed successfully!"

