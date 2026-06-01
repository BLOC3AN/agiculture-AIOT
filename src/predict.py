import sys
from pathlib import Path
from typing import Dict, Any, Union

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.utils import setup_logger
from src.data_loader import Preprocessor
from src.models import HealthEvaluator, IrrigationAdvisor, SoilBalancer

logger = setup_logger("InferencePipeline")

class AgricultureAIoTPipeline:
    """End-to-end inference pipeline for Agriculture AIoT recommendations.
    Loads pre-trained models and preprocesses sensor readings to output advice.
    """
    
    def __init__(self):
        self.model_dir = Config.get_model_absolute_dir()
        self.preprocessor = None
        self.health_evaluator = None
        self.irrigation_advisor = None
        self.soil_balancer = None
        self.is_loaded = False
        
        # Load the models upon initialization
        self.load_pipeline()
        
    def load_pipeline(self) -> None:
        """Loads preprocessor and all machine learning models from disk."""
        try:
            logger.info(f"Loading inference pipeline from directory: {self.model_dir}")
            
            # Load preprocessor
            preprocessor_path = self.model_dir / "preprocessor.pkl"
            if not preprocessor_path.exists():
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}. Run training first.")
            self.preprocessor = Preprocessor.load(preprocessor_path)
            
            # Load Health Evaluator
            self.health_evaluator = HealthEvaluator()
            self.health_evaluator.load(self.model_dir)
            
            # Load Irrigation Advisor
            self.irrigation_advisor = IrrigationAdvisor()
            self.irrigation_advisor.load(self.model_dir)
            
            # Load Soil Balancer
            self.soil_balancer = SoilBalancer()
            self.soil_balancer.load(self.model_dir)
            
            self.is_loaded = True
            logger.info("Agriculture AIoT Pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Agriculture AIoT Pipeline: {e}", exc_info=True)
            self.is_loaded = False
            raise e

    def predict_recommendations(self, sensor_reading: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single sensor reading dictionary and outputs recommendations.
        
        Args:
            sensor_reading: Dictionary containing sensor fields:
                - crop_type: str (e.g. 'Wheat', 'Soybean')
                - soil_moisture_%: float
                - soil_pH: float
                - temperature_C: float
                - humidity_%: float
                - rainfall_mm: float
                - sowing_date: str (e.g. '2024-01-08')
                - timestamp: str (e.g. '2024-03-19')
                - total_days: int
                - NDVI_index: float
                - crop_disease_status: str (e.g. 'None', 'Mild')
                
        Returns:
            Dictionary containing:
                - health_score: float
                - needs_watering: bool
                - water_volume_needed_mm: float
                - needs_ph_balance: bool
                - ph_balance_amount_kg_ha: float
                - soil_status: str
                - advice: str
        """
        if not self.is_loaded:
            logger.error("Pipeline is not loaded. Cannot run inference.")
            raise RuntimeError("Pipeline is not loaded. Call load_pipeline first.")
            
        try:
            logger.info(f"Received sensor reading for crop type: {sensor_reading.get('crop_type')}")
            
            # 1. Transform single reading to feature matrix row
            X_row = self.preprocessor.transform_single(sensor_reading)
            
            # 2. Run Health Evaluator
            health_res = self.health_evaluator.predict(X_row)
            pred_health_score = health_res["predicted_health_score"][0]
            
            # 3. Run Irrigation Advisor
            irrigation_res = self.irrigation_advisor.predict(X_row)
            needs_watering = bool(irrigation_res["needs_watering"][0])
            water_volume = irrigation_res["water_volume_needed_mm"][0]
            
            # 4. Run Soil Balancer
            soil_res = self.soil_balancer.predict(X_row)
            needs_ph_balance = bool(soil_res["needs_ph_balance"][0])
            ph_balance_amount = soil_res["ph_balance_amount_kg_ha"][0]
            
            # 5. Formulate Advice
            soil_ph = float(sensor_reading.get("soil_pH", 6.5))
            soil_status = "Bình thường / Tối ưu"
            if soil_ph < 6.0:
                soil_status = "Đất chua (Axit)"
            elif soil_ph > 7.0:
                soil_status = "Đất kiềm (Bazo)"
                
            advice_list = []
            
            # Health advice
            if pred_health_score > 80:
                advice_list.append("Cây trồng phát triển rất tốt. Duy trì chế độ chăm sóc hiện tại.")
            elif pred_health_score > 50:
                advice_list.append("Cây trồng khỏe mạnh bình thường. Cần theo dõi thêm các dấu hiệu bệnh nhẹ.")
            else:
                disease = sensor_reading.get("crop_disease_status", "None")
                advice_list.append(f"Cảnh báo: Sức khỏe cây kém. Tình trạng bệnh hiện tại: {disease}. Cần can thiệp thuốc bảo vệ thực vật.")
                
            # Irrigation advice
            if needs_watering:
                advice_list.append(f"Khuyến nghị tưới nước: Cần bổ sung ngay lượng nước {water_volume:.2f} mm để đạt độ ẩm tối ưu.")
            else:
                advice_list.append("Độ ẩm đất ổn định, không cần tưới thêm nước.")
                
            # Soil pH advice
            if needs_ph_balance:
                fertilizer_action = "bón vôi" if soil_ph < 6.0 else "bón khoáng chất/sulfur để hạ kiềm"
                advice_list.append(f"Đất cần xử lý pH ({soil_status}): Hãy thực hiện {fertilizer_action} với lượng {ph_balance_amount:.2f} kg/hectare.")
            else:
                advice_list.append("Độ pH đất ở mức tối ưu cho cây trồng. Không cần bón thêm chất cân bằng pH.")
                
            recommendations = {
                "health_score": round(pred_health_score, 2),
                "needs_watering": needs_watering,
                "water_volume_needed_mm": round(water_volume, 2),
                "needs_ph_balance": needs_ph_balance,
                "ph_balance_amount_kg_ha": round(ph_balance_amount, 2),
                "soil_status": soil_status,
                "advice": " | ".join(advice_list)
            }
            
            logger.info("Successfully generated recommendations.")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error predicting recommendations: {e}", exc_info=True)
            raise e

if __name__ == "__main__":
    # Test script if run directly
    try:
        pipeline = AgricultureAIoTPipeline()
        test_reading = {
            "crop_type": "Wheat",
            "soil_moisture_%": 18.5,
            "soil_pH": 5.8,
            "temperature_C": 28.0,
            "humidity_%": 55.0,
            "rainfall_mm": 5.0,
            "sowing_date": "2024-01-08",
            "timestamp": "2024-03-19",
            "total_days": 120,
            "NDVI_index": 0.65,
            "crop_disease_status": "None"
        }
        res = pipeline.predict_recommendations(test_reading)
        print("Test prediction output:")
        import json
        print(json.dumps(res, indent=4, ensure_ascii=False))
    except Exception as ex:
        print(f"Error running standalone test: {ex}")
