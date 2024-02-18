from flask import Flask
from flask import request
from flask import jsonify
from predict import predict

app = Flask('churn')


@app.route("/churn/predict", methods=['POST'])
def churn_predict():
    customer = request.get_json()
    return jsonify(predict(input_customer=customer))


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
