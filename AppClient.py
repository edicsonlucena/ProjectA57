# Databricks notebook source
pip install mlflow

# COMMAND ----------

# Import the dataset from scikit-learn and create the training and test datasets. 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import pandas as pd

db = load_boston()

boston = pd.DataFrame(db.data, columns=db.feature_names)

X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=5)
boston.head()

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-2058505122667366.6.azuredatabricks.net/model/boston-housing-model/3/invocations'
  headers = {'Authorization': f'Bearer dapib8c436d0b077f80e55855a1a5ee734d6'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

score_model(X_test)

# COMMAND ----------

X_test

# COMMAND ----------


