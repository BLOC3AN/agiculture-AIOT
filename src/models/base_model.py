from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Dict, Any

class BaseAgricultureModel(ABC):
    """Abstract base class for all machine learning models in Agriculture AIoT."""
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, train_targets: Dict[str, pd.Series]) -> None:
        """Trains the model(s) on the training dataset."""
        pass
        
    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, test_targets: Dict[str, pd.Series]) -> Dict[str, float]:
        """Evaluates the model(s) on the test dataset and returns performance metrics."""
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Runs predictions on the input features and returns structured recommendations."""
        pass
        
    @abstractmethod
    def save(self, directory: Path) -> None:
        """Saves the trained model state and objects to disk."""
        pass
        
    @abstractmethod
    def load(self, directory: Path) -> None:
        """Loads the trained model state and objects from disk."""
        pass
