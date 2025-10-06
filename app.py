import os
import sys

from flask import Flask, request, render_template

import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/overview')
def project_overview():
    return render_template('overview.html')

@app.route('/predictdata', methods= ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            FlightNumber=int(request.form.get("FlightNumber")),
            Date=request.form.get("Date"),
            BoosterVersion=request.form.get("BoosterVersion"),
            PayloadMass=float(request.form.get("PayloadMass")),
            Orbit=request.form.get("Orbit"),
            LaunchSite=request.form.get("LaunchSite"),
            Flights=int(request.form.get("Flights")),
            GridFins=True if request.form.get("GridFins") == "True" else False,
            Reused=True if request.form.get("Reused") == "True" else False,
            Legs=True if request.form.get("Legs") == "True" else False,
            LandingPad=request.form.get("LandingPad"),
            Block=request.form.get("Block"),
            ReusedCount=int(request.form.get("ReusedCount")),
            Longitude=float(request.form.get("Longitude")),
            Latitude=float(request.form.get("Latitude"))
        )

        input_df = data.get_data_as_data_frame()
        print(input_df)

        predict_pipeline = PredictPipeline()
        outcome = predict_pipeline.predict(input_df)
        print(outcome)

        results = 'Success' if outcome[0] == 1 else "Failure"
        print(results)

        return render_template('home.html', results= results)
    

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug= True)
