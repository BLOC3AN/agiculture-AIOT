import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, Any

from src.models.base_model import BaseAgricultureModel
from src.config import Config
from src.utils import setup_logger

logger = setup_logger("IrrigationAdvisor")

class IrrigationAdvisor(BaseAgricultureModel):
    """Cascaded irrigation decision system:
    1. Classifier determines if watering is needed.
    2. If needed, Regressor predicts the water volume (mm) required.
    """
    
    def __init__(self):
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        self.regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        self.clf_target = "needs_watering"
        self.reg_target = "water_volume_needed"
        
    def train(self, X_train: pd.DataFrame, train_targets: Dict[str, pd.Series]) -> None:
        """Huấn luyện bộ phân loại và bộ hồi quy tưới tiêu."""
        try:
            logger.info(f"Huấn luyện bộ phân loại {self.__class__.__name__}...")
            y_train_clf = train_targets[self.clf_target]
            self.classifier.fit(X_train, y_train_clf)
            
            # Huấn luyện bộ hồi quy chỉ trên những mẫu thực sự cần tưới nước (y == 1)
            logger.info(f"Huấn luyện bộ hồi quy {self.__class__.__name__}...")
            y_train_reg = train_targets[self.reg_target]
            
            # Lọc các dòng cần tưới
            watering_indices = y_train_clf == 1
            X_train_filtered = X_train[watering_indices]
            y_train_reg_filtered = y_train_reg[watering_indices]
            
            if len(X_train_filtered) > 0:
                self.regressor.fit(X_train_filtered, y_train_reg_filtered)
                logger.info(f"Đã huấn luyện bộ hồi quy tưới tiêu trên {len(X_train_filtered)} mẫu.")
            else:
                logger.warning("Không có mẫu nào cần tưới nước trong tập train. Huấn luyện bộ hồi quy trên toàn bộ tập dữ liệu để tránh lỗi.")
                self.regressor.fit(X_train, y_train_reg)
                
            logger.info("Huấn luyện hệ thống tưới tiêu thành công.")
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình Tưới tiêu: {e}", exc_info=True)
            raise e
            
    def evaluate(self, X_test: pd.DataFrame, test_targets: Dict[str, pd.Series]) -> Dict[str, float]:
        """Đánh giá mô hình trên tập test."""
        try:
            logger.info("Đánh giá hệ thống Tưới tiêu...")
            y_test_clf = test_targets[self.clf_target]
            y_test_reg = test_targets[self.reg_target]
            
            # 1. Đánh giá bộ phân loại
            y_pred_clf = self.classifier.predict(X_test)
            acc = accuracy_score(y_test_clf, y_pred_clf)
            prec = precision_score(y_test_clf, y_pred_clf, zero_division=0)
            rec = recall_score(y_test_clf, y_pred_clf, zero_division=0)
            f1 = f1_score(y_test_clf, y_pred_clf, zero_division=0)
            
            logger.info(f"Kết quả phân loại Tưới tiêu - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
            
            # 2. Đánh giá bộ hồi quy (dự đoán theo cơ chế cascaded)
            y_pred_reg = np.zeros(len(X_test))
            
            # Chỉ chạy hồi quy cho những mẫu được phân loại là cần tưới (y_pred_clf == 1)
            need_water_indices = y_pred_clf == 1
            if np.sum(need_water_indices) > 0:
                X_test_filtered = X_test[need_water_indices]
                preds_water = self.regressor.predict(X_test_filtered)
                y_pred_reg[need_water_indices] = preds_water
                
            # Đảm bảo lượng nước không âm
            y_pred_reg = np.clip(y_pred_reg, 0.0, None)
            
            mae = mean_absolute_error(y_test_reg, y_pred_reg)
            mse = mean_squared_error(y_test_reg, y_pred_reg)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_reg, y_pred_reg)
            
            logger.info(f"Kết quả hồi quy Tưới tiêu (Tổng thể) - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            
            # Đánh giá riêng bộ hồi quy trên những mẫu thực sự cần tưới trong tập test
            actual_need_water = y_test_clf == 1
            if np.sum(actual_need_water) > 0:
                mae_filtered = mean_absolute_error(y_test_reg[actual_need_water], y_pred_reg[actual_need_water])
                logger.info(f"Kết quả hồi quy Tưới tiêu (Chỉ các mẫu cần tưới thực tế) - MAE: {mae_filtered:.4f}")
            else:
                mae_filtered = 0.0
                
            return {
                "irrigation_clf_accuracy": float(acc),
                "irrigation_clf_f1": float(f1),
                "irrigation_reg_mae": float(mae),
                "irrigation_reg_rmse": float(rmse),
                "irrigation_reg_r2": float(r2),
                "irrigation_reg_need_water_mae": float(mae_filtered)
            }
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá hệ thống Tưới tiêu: {e}", exc_info=True)
            raise e
            
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Chạy dự đoán để đưa ra quyết định tưới nước."""
        try:
            # 1. Phân loại có cần tưới nước không
            pred_clf = self.classifier.predict(X)
            
            # 2. Tính lượng nước nếu cần
            pred_reg = np.zeros(len(X))
            need_water_indices = pred_clf == 1
            
            if np.sum(need_water_indices) > 0:
                X_filtered = X[need_water_indices]
                preds_water = self.regressor.predict(X_filtered)
                # Đảm bảo lượng nước không âm
                pred_reg[need_water_indices] = np.clip(preds_water, 0.0, None)
                
            return {
                "needs_watering": [int(val) for val in pred_clf],
                "water_volume_needed_mm": [float(val) for val in pred_reg]
            }
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán Tưới tiêu: {e}", exc_info=True)
            raise e
            
    def save(self, directory: Path) -> None:
        """Lưu cả bộ phân loại và bộ hồi quy xuống đĩa."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            clf_path = directory / "irrigation_classifier.pkl"
            reg_path = directory / "irrigation_regressor.pkl"
            joblib.dump(self.classifier, clf_path)
            joblib.dump(self.regressor, reg_path)
            logger.info(f"Đã lưu mô hình Tưới tiêu vào {directory}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình Tưới tiêu: {e}", exc_info=True)
            raise e
            
    def load(self, directory: Path) -> None:
        """Tải mô hình từ đĩa."""
        try:
            clf_path = directory / "irrigation_classifier.pkl"
            reg_path = directory / "irrigation_regressor.pkl"
            self.classifier = joblib.load(clf_path)
            self.regressor = joblib.load(reg_path)
            logger.info(f"Đã tải mô hình Tưới tiêu từ {directory}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Tưới tiêu: {e}", exc_info=True)
            raise e
