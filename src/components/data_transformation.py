import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformer_object(self):
        try:
            num_columns = ['FlightNumber', 'PayloadMass', 'Flights', 'Block', 'ReusedCount', 'Longitude', 'Latitude', 'Year', 'Month', 'DayOfWeek']
            cat_columns = ['BoosterVersion', 'Orbit', 'LaunchSite', 'LandingPad', 'Serial']
            bool_columns = ['GridFins', 'Reused', 'Legs']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= 'median')),
                    ("standard_scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= 'most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown= 'ignore'))
                ]
            )

            bool_pipeline = Pipeline(
                steps= [
                    ("to_int", FunctionTransformer(np.int32)),
                    ("imputer", SimpleImputer(strategy= 'most_frequent'))
                ]
            )

            payload_pipeline = Pipeline(
                steps= [
                    ("imputer", KNNImputer(n_neighbors= 5)),
                    ("scaler", StandardScaler())
                ]
            )

            landingpad_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy= 'constant', fill_value= 'No Pad')),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown= 'ignore'))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_transformer", num_pipeline, [col for col in num_columns if col != 'PayloadMass']),
                    ("cat_transformer", cat_pipeline, [col for col in cat_columns if col != 'LandingPad']),
                    ("bool_transformer", bool_pipeline, bool_columns),
                    ("payload_transformer", payload_pipeline, ['PayloadMass']),
                    ("landingpad_transformer", landingpad_pipeline, ['LandingPad'])
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    

    def preprocess_date(self, df: pd.DataFrame):
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors= 'coerce')
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df = df.drop(['Date'], axis = 1)

            return df

        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation process")

            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info("Preprocessing date")
            train_df = self.preprocess_date(train_df)
            test_df = self.preprocess_date(test_df)


            logging.info("Separating input features and target")
            target_column_name = "Outcome"

            input_feature_train_df = train_df.drop(target_column_name, axis= 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name, axis= 1)
            target_feature_test_df = test_df[target_column_name]

            
            logging.info("Applying preprocessor on training and test dataframe")
            preprocessor_obj = self.get_data_transformer_object()

            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df).toarray()
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df).toarray()
            

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]


            logging.info("Saving preprocessor object to artifacts")
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
