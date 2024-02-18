#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print("Loading data ...")
df = pd.read_csv('data.csv')

C = 1
output_file = f"model_C={C}.bin"

# Data cleaning

print("Data cleaning ...")

df.columns = df.columns.str.lower().str.replace(' ', '_')

str_cols = list(df.dtypes[df.dtypes == 'object'].keys())
str_cols.remove('customerid')
for col in str_cols:
    df[col] = df[col].str.lower().str.replace(" ", "_")

df['churn'] = (df['churn'] == 'yes').astype(int)
df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)
df.totalcharges.isna().sum()

numeric_cols = ['tenure', 'monthlycharges', 'totalcharges']
categorical_cols = ['gender', 'seniorcitizen', 'partner', 'dependents',
                    'phoneservice', 'multiplelines', 'internetservice',
                    'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
                    'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod']

print("Data Splitting ...")

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

dicts = df_train[categorical_cols + numeric_cols].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(dicts)

X_train = dv.transform(df_train[categorical_cols + numeric_cols].to_dict(orient='records'))
y_train = df_train.churn

X_val = dv.transform(df_val[categorical_cols + numeric_cols].to_dict(orient='records'))
y_val = df_val.churn

X_test = dv.transform(df_test[categorical_cols + numeric_cols].to_dict(orient='records'))
y_test = df_test.churn


# Model training

print(f"Model training with C = {C}...")


def train(df, y_train, C=1.0):
    dicts = df[categorical_cols + numeric_cols].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(X_train, y_train)

    return dv, model


def predict(df: pd.DataFrame, dv: DictVectorizer, model: LogisticRegression):
    dicts = df[categorical_cols + numeric_cols].to_dict(orient='records')

    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]


dv, model = train(df_train_full, df_train_full.churn, C)

y_pred = predict(df_test, dv, model)

print(f"ROC AUC score : {roc_auc_score(y_test, y_pred)}")

# Save the model

print("Saving model ...")

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Done.")
