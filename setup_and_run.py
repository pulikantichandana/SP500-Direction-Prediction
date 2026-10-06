import os
import sys
import subprocess
import importlib
import time

# Load API keys from a local .env file (not committed to git) if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _value = _line.partition("=")
            os.environ.setdefault(_key.strip(), _value.strip())

_required_keys = ["FRED_API_KEY", "REDDIT_CLIENT_SECRET", "GEMINI_API_KEY", "REDDIT_CLIENT_ID"]
_missing_keys = [k for k in _required_keys if not os.environ.get(k)]
if _missing_keys:
    print(f"Warning: missing API keys {_missing_keys}. Set them in a local .env file or your environment.")
else:
    print("API keys loaded from environment")

def setup_and_run_model():
    """
    Set up the necessary environment for the model and run it
    """
    print("Setting up and running the improved S&P 500 model...")
    
    # Check and install required packages
    required_packages = [
        "numpy",
        "pandas",
        "tensorflow",
        "sklearn",
        "matplotlib",
        "yfinance",
        "requests",
        "bs4",
        "torch",
        "transformers",
        "praw",
        "google-generativeai",
        "fredapi"
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
            print(f"[OK] {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Import and run the model
    try:
        import improved_sp500_model
        # Force reload to avoid cached imports
        importlib.reload(improved_sp500_model)
        # Run the model - now returns a dict of models
        models, data = improved_sp500_model.train_and_evaluate_model()
        # The ensemble prediction is already done in train_and_evaluate_model
        # No need to call predict_tomorrow_price separately
        # Ensure that the model runs fully by adding a short delay
        time.sleep(1)
        return True
    except Exception as e:
        print(f"Error running the model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    setup_and_run_model() 