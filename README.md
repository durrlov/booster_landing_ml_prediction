# End-to-End Machine Learning Project — SpaceX Falcon 9 Landing Prediction
**Developed by**: [S M Bakhtiar](https://www.linkedin.com/in/durrlov/)  
bakhtiar.scr@gmail.com <br>
[LinkedIn](https://www.linkedin.com/in/durrlov/)

## Important Project Links
### [Try Real-Time Predictions 🔗](http://127.0.0.1:5000/)

[Exploratory Data Analysis Notebook 🔗](https://github.com/durrlov/booster_landing_ml_prediction/blob/main/notebook/03_eda.ipynb)

## Table of contents
- [1. Introduction](#introduction)
- [2. Problem Statement](#problem)
- [3. Dataset Overview](#dataset)
- [4. Exploratory Data Analysis (EDA)](#eda)
- [5. Data Preprocessing & Feature Engineering](#preprocessing)
- [6. Model Selection & Training](#model)
- [7. Model Evaluation](#evaluation)
- [8. Conclusion & Insights](#conclusion)
- [9. Deployment](#deployment)
- [10. Modular Project Architecture & Engineering Practices](#modular)
    - [10.1. Components Breakdown](#components)
    - [10.2. Utility and Support Modules](#support)
    - [10.3. Configuration & Packaging](#configuration)
- [11. Summary](#summary)


## 1. Introduction <a name= "introduction"></a>
This project showcases an **end-to-end machine learning pipeline** designed to predict whether the **first stage (booster) of a SpaceX Falcon 9 rocket** will successfully land.  

This integrates 
- Data collection from SpaceX API
- Data cleaning and feature engineering
- Exploratory Data Analysis (EDA)
- Model training with multiple ML algorithms
- Hyperparameter tuning and model selection
- Flask app for interactive user prediction
- Deployment on Render

## 2. Problem Statement<a name= "problem"></a>
SpaceX advertises Falcon 9 rocket launches with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is because SpaceX can reuse the first stage (booster).

Therefore we aim to determine whether the first stage will land successfully or not. This information can be used if an alternate company wants to bid against SpaceX for a rocket launch.

## 3. Dataset Overview<a name= "dataset"></a>
- Source: 
<br> Collected data from SpaceX REST API's:
    - https://api.spacexdata.com/v4/launches/past/
    - https://api.spacexdata.com/v4/rockets/
    - https://api.spacexdata.com/v4/launchpads/
    - https://api.spacexdata.com/v4/payloads/
    - https://api.spacexdata.com/v4/cores/

- Target Variable:
<br>Outcome : Preprocessed to binary classification
    - Success = 1
    - Failure = 0

- Features:

| Column          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| FlightNumber    | Sequential number assigned to each SpaceX launch.                           |
| Date            | Launch date of the mission.                                                 |
| BoosterVersion  | Specific Falcon 9 booster version used for the mission.                     |
| PayloadMass     | Mass of the payload carried by the rocket (in kilograms).                   |
| Orbit           | Type of orbit the payload was delivered to (e.g., LEO, GTO).                |
| LaunchSite      | Location from where the rocket was launched.                                |
| Flights         | Total missions for this booster (including the current one)                 |
| GridFins        | Whether grid fins were present on the booster (True/False).                 |
| Reused          | Whether the booster was previously reused (True/False).                     |
| Legs            | Whether the booster had landing legs (True/False).                          |
| LandingPad      | Landing location of the booster (e.g., drone ship, ground pad).             |
| Block           | Version block of the Falcon 9 booster, indicating design upgrades.          |
| ReusedCount     | Number of times the booster had been reused before the flight.              |
| Serial          | Unique identifier for the rocket booster.                                   |
| Longitude       | Longitude coordinate of the launch site.                                    |
| Latitude        | Latitude coordinate of the launch site.                                     |

- Rows: 168 Launches

- Missing Values:
    - PayloadMass 22
    - LandingPad 26
    - Orbit 1

## 4. Exploratory Data Analysis (EDA)<a name= "eda"></a>
- Univariate Analysis:
    - Barchart for different categorical and boolean features
    - Histogram, Boxplot, and Violinplot for different numerical features
- Bivariate Analysis:
    - Landing Success rate for different categorical and boolean features
    - Pairwise relationships among different categorical and numerical features
- Correlation table and correlation heatmap
- Temporal trends of successful landings

## 5. Data Preprocessing & Feature Engineering<a name= "preprocessing"></a>
- Categorical Encoding: OneHotEncoder for all categorical features
- Boolean Transformation: Converted boolean values (True/False) to numeric (1/0)
- Feature Scaling: StandardScaler for numerical features
- Pipeline: ColumnTransformer used to combine transformations
- Artifacts Saved: Preprocessor pipeline saved using pickle (.pkl) for reuse during inference

## 6. Model Selection & Training<a name= "model"></a>
- Models evaluated:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Random Forest
    - AdaBoost
    - Gradient Boosting
    - Support Vector Machine
    - XGBoost
    - CatBoost
- Hyperparameter Tuning: GridSearchCV
- Train-Test Split: Used train_test_split() with appropriate random seed
- Artifacts Saved: Best model saved for deployment

## 7. Model Evaluation<a name= "evaluation"></a>
- Metrics Used:
    - Accuracy Score
    - Jaccard Score
    - Precision  
    - Recall  
    - F1-Score  
    - ROC-AUC
- Best model:
    - The XGBoost model achieved highest accuracy score (~ 88.24%) on test data
- The best model's predictions were evaluated via 
    - Confusion Matrix
    - ROC Curve
    - Classification Report

## 8. Conclusion & Insights<a name= "conclusion"></a>
- Launch to VLEO, LEO and ISS orbits have a very high success rate to other orbits, suggesting others are more challenging for re-landing the first stage.
- KSC LC 39A and VAFB SLC 4E launch sites have a much higher success rate compared to CCSFS SLC 40, indicating the launch location is a key factor in landing success.
- High number of successful landings occurred when components like GridFins, and Legs were used, confirming their importance in the recovery process.
- Without landing pad, the success rate is very low.
- The more the payload mass, the more likely the first stage will land successfully.
- Temporal trends shows a significant upward trend in the landing success rate over the years.

## 9. Deployment<a name= "deployment"></a>
- Created a Flask app to serve predictions
- CustomData class used to collect and format user input
- PredictPipeline class used to load saved model and preprocessor
- Simple HTML form captures user data
- Automatically fills launch site coordinates (latitude, longitude) based on selection.
- Hosted on Render with automatic deployment from GitHub

## 10. Modular Project Architecture & Engineering Practices<a name= "modular"></a>
This project maintains industry standards by following a modular and scalable architecture:

#### 10.1. Components Breakdown<a name= "components"></a>
- data_ingestion.py  
  Handles reading raw data from source (e.g., CSV), splitting it into training and testing sets, and saving these artifacts in the artifacts directory.

- data_transformation.py  
  Builds preprocessing pipelines using ColumnTransformer, handles categorical and numerical processing, and serializes the preprocessor object for later inference.

- model_trainer.py  
  Encapsulates model training, evaluation, and hyperparameter tuning. Supports multiple regression algorithms and exports the best-performing model.

- train_pipeline.py  
  Runs the full model training workflow by calling data ingestion, transformation, and model trainer modules.

- predict_pipeline.py 
  Loads the saved model and preprocessor to make predictions on new user input collected via the web form.

#### 10.2. Utility and Support Modules<a name= "support"></a>
- logger.py  
  Custom logging setup to track pipeline progress, errors, and debugging information.

- exception.py  
  A custom exception class to wrap and raise errors cleanly across modules for better traceability.

- utils.py  
  Includes reusable functions like saving and loading objects, evaluating models, and other helper operations.

#### 10.3. Configuration & Packaging<a name= "configuration"></a>
- setup.py  
  Makes the project installable as a package. Useful for deployment and future integrations.

- requirements.txt  
  Lists all Python dependencies used in the project.

## 11. Summary<a name= "summary"></a>
This project demonstrates a robust, industry-standard machine learning pipeline from data collection from API to deployment. It combines solid EDA, modular engineering practices, and model explainability, making it suitable for both real-world use and demonstrating technical skill in  professional settings.