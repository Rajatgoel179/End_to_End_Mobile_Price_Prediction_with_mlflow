import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load("/workspaces/End_to_End_Mobile_Price_Prediction_with_mlflow/artifacts/model_trainer/model.joblib")

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction