import os
import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object, preprocess_date

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path= model_path)
            preprocessor = load_object(file_path= preprocessor_path)

            data = preprocess_date(df= features)
            data_scaled = preprocessor.transform(data)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 FlightNumber: int,
                 Date: str,
                 BoosterVersion: str,
                 PayloadMass: float,
                 Orbit: str,
                 LaunchSite: str,
                 Flights: int,
                 GridFins: bool,
                 Reused: bool,
                 Legs: bool,
                 LandingPad: str,
                 Block: str,
                 ReusedCount: int,
                 Longitude: float,
                 Latitude: float):
        self.FlightNumber = FlightNumber
        self.Date = Date
        self.BoosterVersion = BoosterVersion
        self.PayloadMass = PayloadMass
        self.Orbit = Orbit
        self.LaunchSite = LaunchSite
        self.Flights = Flights
        self.GridFins = GridFins
        self.Reused = Reused
        self.Legs = Legs
        self.LandingPad = LandingPad
        self.Block = Block
        self.ReusedCount = ReusedCount
        self.Longitude = Longitude
        self.Latitude = Latitude


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict ={
                "FlightNumber": [self.FlightNumber],
                "Date": [self.Date],
                "BoosterVersion": [self.BoosterVersion],
                "PayloadMass": [self.PayloadMass],
                "Orbit": [self.Orbit],
                "LaunchSite": [self.LaunchSite],
                "Flights": [self.Flights],
                "GridFins": [self.GridFins],
                "Reused": [self.Reused],
                "Legs": [self.Legs],
                "LandingPad": [self.LandingPad],
                "Block": [self.Block],
                "ReusedCount": [self.ReusedCount],
                "Longitude": [self.Longitude],
                "Latitude": [self.Latitude]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)