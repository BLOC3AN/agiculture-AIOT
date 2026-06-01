import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    """Configuration class for the Agriculture AIoT system.
    Loads configurations from .env with default fallbacks.
    """
    WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

    # Paths
    DATA_PATH = os.getenv("DATA_PATH", "data/Smart_Farming_Crop_Yield_2024.csv")
    MODEL_DIR = os.getenv("MODEL_DIR", "models/")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Machine Learning Parameters
    RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

    # Crop Cycle Mapping Config
    _DEFAULT_MAPPING = '{"Wheat": 120.0, "Soybean": 120.0, "Cotton": 120.0, "Rice": 120.0, "Maize": 120.0, "Tomato": 75.0, "Pepper": 85.0, "Lettuce": 40.0, "Cucumber": 60.0}'
    CROP_CYCLE_MAPPING = json.loads(os.getenv("CROP_CYCLE_MAPPING", _DEFAULT_MAPPING))

    @classmethod
    def get_data_absolute_path(cls) -> Path:
        """Returns the absolute path to the dataset."""
        path = Path(cls.DATA_PATH)
        if not path.is_absolute():
            return cls.WORKSPACE_ROOT / path
        return path

    @classmethod
    def get_model_absolute_dir(cls) -> Path:
        """Returns the absolute path to the model storage directory."""
        path = Path(cls.MODEL_DIR)
        if not path.is_absolute():
            return cls.WORKSPACE_ROOT / path
        return path
