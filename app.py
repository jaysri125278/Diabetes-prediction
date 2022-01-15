from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('diabetes.pkl','rb'))

@app.route('/')
def Home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    X = [float(x) for x in request.form.values()]
    features = [np.array(X)]
    prediction = model.predict(features)

    if prediction == 1:
        return render_template('home.html', prediction_text="You have diabetes")

    else:
        return render_template('home.html', prediction_text="You don't have diabetes")

    

if __name__ == "__main__":
    app.run(debug=True)