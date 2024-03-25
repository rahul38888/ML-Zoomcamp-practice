import pickle

with open('model_C=1.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def predict(input_customer: dict):
    X = dv.transform([input_customer])
    churn_prob = model.predict_proba(X)[0, 1]
    y_pred = (churn_prob < 0.5)

    return {"Probability": churn_prob, "Prediction": "Will Churn" if y_pred else "Will not churn"}
