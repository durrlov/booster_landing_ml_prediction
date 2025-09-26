import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig

    
    def get_data_transformer_object(self):
        try:
            cat_column = []
            num_column = []

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= 'constant', fill_value= 'No Pad')),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown= 'ignore'))
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", KNNImputer(n_neighbors= 7)),
                    ("standard_scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("cat_transformer", cat_pipeline(), cat_column),
                    ("num_transformer", num_pipeline(), num_column)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            pass

        except Exception as e:
            pass
