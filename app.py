from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Brand =int(request.form['Brand'])
            Storage =int(request.form['Storage'])
            RAM =int(request.form['RAM'])
            Battery =int(request.form['Battery'])
            n_cameras =int(request.form['n_cameras'])
            res1 =int(request.form['res1'])
            res2 =int(request.form['res2'])
            res3 =int(request.form['res3'])
            res4 =int(request.form['res4'])
            screen =float(request.form['screen'])
       
         
            data = [[Brand,Storage,RAM,Battery,n_cameras,res1,res2,res3,res4,screen]]
            data = np.array(data).reshape(1, 10)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = predict )

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)