import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, evaluate_models

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    grid_report_file_path: str = os.path.join('artifacts', 'grid_report.csv')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info(f"Starting model trainging process")

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'KNN': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'SVM': SVC(probability=True),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbose=0),
                'CatBoost': CatBoostClassifier(verbose=0, train_dir=None )
            }

            params = {
                'Logistic Regression': {
                    'C': np.logspace(-4, 4, 9),
                    'solver': ['liblinear', 'lbfgs']
                },
                'Decision Tree': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [3, 5, 8, 10, 12, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'Random Forest': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.5, 1]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.7, 0.85, 1.0],
                    'max_depth': [3, 5, 8]
                },
                'SVM': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.1, 1]
                },
                'XGBoost': {
                    'n_estimators': [50, 100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 10],
                    'subsample': [0.7, 0.8, 0.9, 1.0],
                    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.5]
                },
                'CatBoost': {
                    'iterations': [100, 200, 500, 1000],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'depth': [3, 5, 7],
                    'l2_leaf_reg': [1, 3, 5, 9]
                }
            }

            best_model, report_df = evaluate_models(
                X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test,
                models= models, params= params
            )

            logging.info(f"Evaluation report")
            logging.info("\n" + report_df.to_string(index= False))

            logging.info("Saving the best model to artifacts")
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            report_df.to_csv(self.model_trainer_config.grid_report_file_path, index= False)

            return best_model.__class__.__name__


        except Exception as e:
            raise CustomException(e, sys)