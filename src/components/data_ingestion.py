import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")

        try:
            df = pd.read_csv("notebook/data/data_from_api.csv")
            logging.info('Dataset has been read as a dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index= False)
            logging.info('Raw data has been saved to artifacts/data.csv')

            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42)
            
            train_set.to_csv(self.data_ingestion_config.train_data_path, index= False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index= False)
            logging.info("Train and test data has been saved to artifacts")

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    logging.info(f"Data ingestion complete. Train data at: {train_path}, Test data at: {test_path}")