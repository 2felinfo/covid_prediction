import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__) #Initialize the flask App
loaded_model = joblib.load("finalized_model.joblib")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    X_test = [float(x) for x in request.form.values()]
    Xx_test=np.array(X_test)
    Xx_test=Xx_test.reshape(1,-1)
    result = loaded_model.predict(Xx_test)
    if result[0]==0:
            infected=""
            return render_template('index.html',safe="You are safe!!")
    else:
        safe=""
        return render_template('index.html',infected="  You are infected :(")

        



    
if __name__ == "__main__":
    app.run(debug=True)
