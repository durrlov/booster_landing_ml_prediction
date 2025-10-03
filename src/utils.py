import os
import sys
import joblib
import json

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, jaccard_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_path)
        
    except Exception as e:
        raise CustomException(e, sys)
    

def preprocess_date(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors= 'coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df = df.drop(['Date'], axis = 1)

        return df

    except Exception as e:
        raise CustomException(e, sys)


def get_metrics(true, predicted, predicted_proba= None):
    try:
        acc = accuracy_score(true, predicted)
        jac = jaccard_score(true, predicted)
        prec = precision_score(true, predicted, average='weighted', zero_division=0)
        rec = recall_score(true, predicted, average='weighted')
        f1 = f1_score(true, predicted, average='weighted')

        metrics = {
            'Accuracy': acc,
            'Jaccard': jac,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        }

        if predicted_proba is not None and len(np.unique(true)) == 2:
            auc = roc_auc_score(true, predicted_proba[:, 1])
            metrics['ROC AUC'] = auc
            
        return metrics

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict):
    try:
        report = []
        best_model = None
        best_score = - np.inf


        for model_name, model in models.items():
            grid = GridSearchCV(
                estimator= model,
                param_grid= params[model_name],
                cv= 5,
                scoring= 'accuracy',
                n_jobs= -1,
                error_score= 'raise'
            )

            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            best_param = grid.best_params_

            y_pred_train= best_estimator.predict(X_train)
            y_pred_test= best_estimator.predict(X_test)
            y_proba_test = best_estimator.predict_proba(X_test) if hasattr(best_estimator, 'predict_proba') else None

            metrics_train = get_metrics(y_train, y_pred_train)
            metrics_test = get_metrics(y_test, y_pred_test, y_proba_test)

            report.append({
                'Model': model_name,
                'Best Params': best_param,
                'Train Accuracy': metrics_train['Accuracy'],
                'Test Accuracy': metrics_test['Accuracy'],
                'Test Jaccard': metrics_test['Jaccard'],
                'Test Precision': metrics_test['Precision'],
                'Test Recall': metrics_test['Recall'],
                'Test F1': metrics_test['F1 Score'],
                'Test ROC AUC': metrics_test.get('ROC AUC', None)
            })


            if metrics_test['Accuracy'] >= best_score:
                best_model = best_estimator
                best_score = metrics_test['Accuracy']

        report_df = pd.DataFrame(report).sort_values(by=['Test Accuracy'], ascending=False)
        return best_model, report_df

    except Exception as e:
        raise CustomException(e, sys)