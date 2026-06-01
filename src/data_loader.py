import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

from src.config import Config
from src.utils import setup_logger

logger = setup_logger("DataLoader")

class Preprocessor:
    """Handles feature engineering and encoding for inputs of all models."""
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.crop_columns = []
        self.is_fitted = False

    def fit(self, df: pl.DataFrame):
        """Fits the one-hot encoder on categorical columns."""
        try:
            logger.info("Fitting preprocessor...")
            crop_types = df.select("crop_type").to_pandas()
            self.encoder.fit(crop_types)
            self.crop_columns = [f"crop_{c}" for c in self.encoder.categories_[0]]
            self.is_fitted = True
            logger.info(f"Preprocessor fitted successfully. Crop categories: {self.encoder.categories_[0]}")
        except Exception as e:
            logger.error(f"Error fitting preprocessor: {e}", exc_info=True)
            raise e

    def transform(self, df: pl.DataFrame) -> pd.DataFrame:
        """Transforms raw DataFrame into a feature matrix (pandas DataFrame)."""
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted yet. Call fit first.")
            
        try:
            # 1. Feature Engineering
            df_processed = df.with_columns([
                # Convert date strings to Date type if they aren't already
                pl.col("sowing_date").str.to_date(),
                pl.col("timestamp").str.to_date(),
            ])
            
            # Auto-calculate/fallback for total_days if missing or null based on crop type
            crop_cycle_mapping = {
                "Wheat": 120.0,
                "Soybean": 120.0,
                "Cotton": 120.0,
                "Rice": 120.0,
                "Maize": 120.0,
                "Tomato": 75.0,
                "Pepper": 85.0,
                "Lettuce": 40.0,
                "Cucumber": 60.0
            }
            if "total_days" not in df_processed.columns:
                df_processed = df_processed.with_columns([
                    pl.col("crop_type").replace(crop_cycle_mapping, default=100.0).cast(pl.Float64).alias("total_days")
                ])
            else:
                df_processed = df_processed.with_columns([
                    pl.col("total_days").fill_null(
                        pl.col("crop_type").replace(crop_cycle_mapping, default=100.0).cast(pl.Float64)
                    )
                ])
            
            # Calculate days since sowing and growth ratio
            df_processed = df_processed.with_columns([
                (pl.col("timestamp") - pl.col("sowing_date")).dt.total_days().alias("days_since_sowing")
            ])
            
            df_processed = df_processed.with_columns([
                (pl.col("days_since_sowing").cast(pl.Float64) / pl.col("total_days").cast(pl.Float64)).alias("growth_ratio")
            ])
            
            # 2. Select numerical features
            num_cols = ["growth_ratio", "soil_moisture_%", "soil_pH", "temperature_C", "humidity_%", "rainfall_mm"]
            df_num = df_processed.select(num_cols).to_pandas()
            
            # 3. Transform categorical features (crop_type)
            crop_types = df_processed.select("crop_type").to_pandas()
            crop_encoded = self.encoder.transform(crop_types)
            df_crop = pd.DataFrame(crop_encoded, columns=self.crop_columns, index=df_num.index)
            
            # 4. Concatenate numerical and encoded categorical features
            X = pd.concat([df_crop, df_num], axis=1)
            
            # Check for any NaNs and fill them just in case
            if X.isnull().values.any():
                logger.warning("Found NaN values in processed features, filling with column means/zeros.")
                X = X.fillna(X.mean().fillna(0))
                
            return X
        except Exception as e:
            logger.error(f"Error transforming features: {e}", exc_info=True)
            raise e

    def transform_single(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Transforms a single input dictionary (from API/Inference) into a feature matrix row."""
        try:
            # Convert single dictionary to Polars DataFrame for processing consistency
            df_single = pl.DataFrame([input_data])
            return self.transform(df_single)
        except Exception as e:
            logger.error(f"Error transforming single input: {e}", exc_info=True)
            raise e

    def save(self, path: Path):
        """Saves the preprocessor state to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, path)
            logger.info(f"Saved preprocessor to {path}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}", exc_info=True)
            raise e

    @staticmethod
    def load(path: Path) -> 'Preprocessor':
        """Loads preprocessor state from disk."""
        try:
            preprocessor = joblib.load(path)
            logger.info(f"Loaded preprocessor from {path}")
            return preprocessor
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}", exc_info=True)
            raise e


class AgricultureDataLoader:
    """Loads datasets, builds targets, and prepares train/test splits."""
    
    def __init__(self):
        self.data_path = Config.get_data_absolute_path()
        self.preprocessor = Preprocessor()

    def load_and_preprocess(self) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
        """Loads raw dataset, engineers targets, and transforms features.
        
        Returns:
            X: Feature matrix
            y_train_targets: Dictionary of target Series for training
            y_test_targets: Dictionary of target Series for testing
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pl.read_csv(self.data_path)
            
            # --- TARGET ENGINEERING ---
            # 1. Health Score (Branch 1)
            disease_penalties = {
                "None": 0.0,
                "Mild": 0.15,
                "Moderate": 0.35,
                "Severe": 0.60
            }
            # Replace disease status with penalty floats
            df = df.with_columns([
                pl.col("crop_disease_status").replace(disease_penalties).cast(pl.Float64).alias("disease_penalty")
            ])
            df = df.with_columns([
                ((1.0 - pl.col("disease_penalty")) * pl.col("NDVI_index") * 100.0).alias("health_score")
            ])
            
            # 2. Irrigation Branch (Branch 2)
            df = df.with_columns([
                pl.when(pl.col("soil_moisture_%") < 25.0).then(1).otherwise(0).alias("needs_watering")
            ])
            
            # Calculate volume needed if watering is required
            df = df.with_columns([
                pl.when(pl.col("needs_watering") == 1)
                .then(
                    # Base formula with adjustments for temperature and rainfall
                    1.2 * (35.0 - pl.col("soil_moisture_%")) +
                    0.5 * (pl.col("temperature_C") - 25.0) -
                    0.2 * pl.col("rainfall_mm")
                )
                .otherwise(0.0)
                .alias("water_volume_needed")
            ])
            # Ensure water volume is not negative
            df = df.with_columns([
                pl.when(pl.col("water_volume_needed") < 0.0)
                .then(0.0)
                .otherwise(pl.col("water_volume_needed"))
                .alias("water_volume_needed")
            ])
            
            # 3. Soil Treatment Branch (Branch 3)
            df = df.with_columns([
                pl.when((pl.col("soil_pH") < 6.0) | (pl.col("soil_pH") > 7.0))
                .then(1)
                .otherwise(0)
                .alias("needs_ph_balance")
            ])
            
            # Calculate amount of lime/mineral needed
            df = df.with_columns([
                pl.when(pl.col("needs_ph_balance") == 1)
                .then(150.0 * (pl.col("soil_pH") - 6.5).abs())
                .otherwise(0.0)
                .alias("ph_balance_amount")
            ])
            
            # --- FEATURE TRANSFORMATION ---
            self.preprocessor.fit(df)
            X = self.preprocessor.transform(df)
            
            # Extract targets as pandas Series
            targets = {
                "health_score": df["health_score"].to_pandas(),
                "needs_watering": df["needs_watering"].to_pandas(),
                "water_volume_needed": df["water_volume_needed"].to_pandas(),
                "needs_ph_balance": df["needs_ph_balance"].to_pandas(),
                "ph_balance_amount": df["ph_balance_amount"].to_pandas(),
            }
            
            logger.info("Data loaded and preprocessed successfully.")
            return X, targets
            
        except Exception as e:
            logger.error(f"Error in load_and_preprocess: {e}", exc_info=True)
            raise e

    def get_splits(self) -> Dict[str, Any]:
        """Loads data and splits into train and test sets for all targets."""
        try:
            X, targets = self.load_and_preprocess()
            
            splits = {
                "X_train": None,
                "X_test": None,
                "train_targets": {},
                "test_targets": {}
            }
            
            # Split features and all targets together to maintain row alignment
            # Using random_state from Config
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE
            )
            
            splits["X_train"] = X.iloc[train_idx]
            splits["X_test"] = X.iloc[test_idx]
            
            for key, target_series in targets.items():
                splits["train_targets"][key] = target_series.iloc[train_idx]
                splits["test_targets"][key] = target_series.iloc[test_idx]
                
            logger.info(f"Train/Test split completed. Train size: {len(train_idx)}, Test size: {len(test_idx)}")
            return splits
        except Exception as e:
            logger.error(f"Error spliting data: {e}", exc_info=True)
            raise e
