import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
import numpy as np
from typing import Dict, Any

from src.models.base_model import BaseAgricultureModel
from src.config import Config
from src.utils import setup_logger

logger = setup_logger("HealthEvaluator")

class HealthEvaluator(BaseAgricultureModel):
    """Regression model to evaluate crop health score (0-100)."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        self.target_name = "health_score"
        
    def train(self, X_train: pd.DataFrame, train_targets: Dict[str, pd.Series]) -> None:
        """Huấn luyện mô hình dự đoán điểm sức khỏe."""
        try:
            logger.info(f"Huấn luyện mô hình {self.__class__.__name__}...")
            y_train = train_targets[self.target_name]
            self.model.fit(X_train, y_train)
            logger.info("Huấn luyện mô hình Đánh giá Sức khỏe thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình Đánh giá Sức khỏe: {e}", exc_info=True)
            raise e
            
    def evaluate(self, X_test: pd.DataFrame, test_targets: Dict[str, pd.Series]) -> Dict[str, float]:
        """Đánh giá chất lượng mô hình trên tập test."""
        try:
            logger.info("Đánh giá mô hình Đánh giá Sức khỏe...")
            y_test = test_targets[self.target_name]
            y_pred = self.model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "health_mae": float(mae),
                "health_rmse": float(rmse),
                "health_r2": float(r2)
            }
            
            logger.info(f"Kết quả Đánh giá Sức khỏe - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            return metrics
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá mô hình Đánh giá Sức khỏe: {e}", exc_info=True)
            raise e
            
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Dự đoán điểm sức khỏe cho dữ liệu đầu vào."""
        try:
            predictions = self.model.predict(X)
            # Clip values between 0 and 100
            predictions = np.clip(predictions, 0.0, 100.0)
            
            return {
                "predicted_health_score": [float(val) for val in predictions]
            }
        except Exception as e:
            logger.error(f"Lỗi khi chạy dự đoán Đánh giá Sức khỏe: {e}", exc_info=True)
            raise e
            
    def save(self, directory: Path) -> None:
        """Lưu mô hình xuống đĩa."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            model_path = directory / "health_evaluator.pkl"
            joblib.dump(self.model, model_path)
            logger.info(f"Đã lưu mô hình Đánh giá Sức khỏe vào {model_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình Đánh giá Sức khỏe: {e}", exc_info=True)
            raise e
            
    def load(self, directory: Path) -> None:
        """Tải mô hình từ đĩa."""
        try:
            model_path = directory / "health_evaluator.pkl"
            self.model = joblib.load(model_path)
            logger.info(f"Đã tải mô hình Đánh giá Sức khỏe từ {model_path}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Đánh giá Sức khỏe: {e}", exc_info=True)
            raise e
