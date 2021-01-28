import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    '''
    For rendering results on HTML GUI
    '''

    features = request.form.to_dict()
    features = list(features.values())
    features = list(map(int, features))
    final_features = np.array(features).reshape(1,6)
    prediction = model.predict(final_features)

    #select = request.form.get('category')
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Furniture prediction price is : $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    '''
    For direct API calls through request
    '''

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)