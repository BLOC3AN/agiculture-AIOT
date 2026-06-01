import sys
from pathlib import Path

# Add root folder to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from src.config import Config
from src.utils import setup_logger
from src.data_loader import AgricultureDataLoader
from src.models import HealthEvaluator, IrrigationAdvisor, SoilBalancer

logger = setup_logger("TrainPipeline")

def main():
    """Main function to train and evaluate all models in the Agriculture AIoT system."""
    try:
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN DỰ ÁN AGRICULTURE AIOT")
        logger.info("=" * 60)
        
        # 1. Khởi tạo DataLoader và tải dữ liệu
        data_loader = AgricultureDataLoader()
        splits = data_loader.get_splits()
        
        X_train = splits["X_train"]
        X_test = splits["X_test"]
        train_targets = splits["train_targets"]
        test_targets = splits["test_targets"]
        
        # 2. Lưu bộ tiền xử lý (Preprocessor) để tái sử dụng khi suy luận
        model_dir = Config.get_model_absolute_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        preprocessor_path = model_dir / "preprocessor.pkl"
        data_loader.preprocessor.save(preprocessor_path)
        
        # 3. Khởi tạo các lớp quản lý mô hình
        health_evaluator = HealthEvaluator()
        irrigation_advisor = IrrigationAdvisor()
        soil_balancer = SoilBalancer()
        
        # 4. Huấn luyện từng mô hình
        logger.info("-" * 40)
        logger.info("Bước 1: Huấn luyện mô hình Đánh giá Sức khỏe...")
        health_evaluator.train(X_train, train_targets)
        
        logger.info("-" * 40)
        logger.info("Bước 2: Huấn luyện mô hình Khuyên dùng Tưới tiêu...")
        irrigation_advisor.train(X_train, train_targets)
        
        logger.info("-" * 40)
        logger.info("Bước 3: Huấn luyện mô hình Cân bằng Đất...")
        soil_balancer.train(X_train, train_targets)
        
        # 5. Đánh giá chất lượng các mô hình trên tập Test
        logger.info("=" * 60)
        logger.info("BẮT ĐẦU ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH")
        logger.info("=" * 60)
        
        metrics = {}
        
        health_metrics = health_evaluator.evaluate(X_test, test_targets)
        metrics.update(health_metrics)
        
        irrigation_metrics = irrigation_advisor.evaluate(X_test, test_targets)
        metrics.update(irrigation_metrics)
        
        soil_metrics = soil_balancer.evaluate(X_test, test_targets)
        metrics.update(soil_metrics)
        
        # In tóm tắt kết quả
        logger.info("-" * 40)
        logger.info("TÓM TẮT CHỈ SỐ ĐÁNH GIÁ:")
        logger.info(json.dumps(metrics, indent=4))
        logger.info("-" * 40)
        
        # Lưu kết quả đánh giá thành file json
        metrics_path = model_dir / "evaluation_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Đã lưu các chỉ số đánh giá vào {metrics_path}")
        
        # 6. Lưu trạng thái các mô hình đã huấn luyện
        logger.info("-" * 40)
        logger.info("Lưu trữ các mô hình xuống thư mục models/...")
        health_evaluator.save(model_dir)
        irrigation_advisor.save(model_dir)
        soil_balancer.save(model_dir)
        
        logger.info("=" * 60)
        logger.info("QUY TRÌNH HUẤN LUYỆN HOÀN THÀNH THÀNH CÔNG VÀ AN TOÀN")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng trong quy trình huấn luyện: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
