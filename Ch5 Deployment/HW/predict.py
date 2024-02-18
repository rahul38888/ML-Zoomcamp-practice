from flask import Flask
from flask import request
from flask import jsonify
import pickle

app = Flask('predict')

def load_model_and_dv():
    model_file = 'model2.bin'
    dv_file = 'dv.bin'
    model = None
    dv = None
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(dv_file, 'rb') as f:
        dv = pickle.load(f)
    
    return model, dv


model, dv = load_model_and_dv()


@app.route("/predict", methods=['POST'])
def predict_proba():
    client = request.get_json()
    X = dv.transform(client)
    return jsonify(model.predict_proba(X)[:,1][0])