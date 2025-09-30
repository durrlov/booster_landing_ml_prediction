import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion complete. Train data at: {train_path}, Test data at: {test_path}")
            logging.info("="*75)

            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
            logging.info(f"Data transformation complete")
            logging.info("="*75)

            trainer = ModelTrainer()
            best_model_name = trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training complete")
            logging.info("="*75)

            logging.info(f"Training pipeline executed successfully. Best model: {best_model_name}")
            logging.info("="*75)

            return best_model_name

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = TrainPipeline()
    best_model_name = obj.run_pipeline()
    print(f"\n Training pipeline executed successfully! Best model: {best_model_name}")