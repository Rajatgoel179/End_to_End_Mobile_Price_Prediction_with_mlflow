from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)  # initializing a flask app

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

# @app.route('/train', methods=['GET'])  # route to train the pipeline
# def training():
#     os.system("python main.py")
#     return "Training Successful!" 

@app.route("/predict", methods=['POST'])
def predict_price():
    if request.method == 'POST':
        Brand = int(request.form["Brand"])
        Storage = int(request.form["Storage"])
        RAM = int(request.form["RAM"])
        Battery = int(request.form["Battery_Capacity"])
        n_cameras = int(request.form["n_cameras"])
        res1 = int(request.form["res1"])
        res2 = int(request.form["res2"])
        res3 = int(request.form["res3"])
        res4 = int(request.form["res4"])
        screen = float(request.form["screen"])
            
            
        data = [Brand, Storage, RAM, Battery, n_cameras, res1, res2, res3, res4, screen]
        data = np.array(data).reshape(1, 10)
            
        obj = PredictionPipeline()
        output = obj.predict(data)

        if output < 0:
            return render_template("home.html", Prediction_text="Sorry")
        else:
            return render_template("home.html", Prediction_text="You buy {}".format(output))

    else:
        return render_template("home.html", Prediction_text="You can sell your car for {}".format(output))
if __name__ == "__main__":
    app.run(debug=True)


