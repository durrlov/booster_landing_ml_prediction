import os
import sys
import joblib
import json

from src.exception import CustomException
from src.logger import logging

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)

        with open(file_path, 'wb') as file_obj:
            joblib.dump(obj, file_path)

    except Exception as e:
        CustomException(e, sys)