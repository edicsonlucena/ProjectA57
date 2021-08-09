# Databricks notebook source
pip install mlflow

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

# MAGIC %md  transition the model from Dev (None) to the Staging stage.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()

model_name = "boston-housing-model" # Replace this with the name of your registered model, if necessary.
latest_version_info = client.get_latest_versions(model_name, stages=["None"])
latest_dev_version = latest_version_info[0].version
print("The latest development version of the model '%s' is '%s'." % (model_name, latest_dev_version))


# COMMAND ----------

import mlflow.pyfunc

model_version_uri = "models:/{model_name}/1".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

model_dev_uri = "models:/{model_name}/None".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_dev_uri))
model_production = mlflow.pyfunc.load_model(model_dev_uri)

# COMMAND ----------

# MAGIC %md ## Prediction prix output with the production model

# COMMAND ----------

# Import the dataset from scikit-learn and create the training and test datasets. 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import mlflow.tensorflow
import pandas as pd

db = load_boston()

boston = pd.DataFrame(db.data, columns=db.feature_names)

X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
boston.head()

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

n_estimators = 100
max_depth = 6
max_features = 3
  # Create and train model
rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)

# COMMAND ----------

def plot(model_name, model_stage, model_version, power_predictions, past_power_output):
  import pandas as pd
  #import matplotlib.dates as mdates
  from matplotlib import pyplot as plt
  index = power_predictions.index
  fig = plt.figure(figsize=(11, 7))
  ax = fig.add_subplot(111)
  ax.set_xlabel("", size=20, labelpad=20)
  ax.set_ylabel("Prix Million", size=20, labelpad=60, rotation=0)
  ax.tick_params(axis='both', which='major', labelsize=17)
  # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
  ax.plot(index[:len(past_power_output)], past_power_output, label="True", color="red", alpha=0.5, linewidth=4)
  ax.plot(index, power_predictions.squeeze(), "--", label="Predicted by '%s'\nin stage '%s' (Version %d)" % (model_name, model_stage, model_version), color="blue", linewidth=3)
  ax.set_ylim(ymin=0, ymax=max(100, int(max(power_predictions.values) * 1.3)))
  ax.legend(fontsize=14)
  plt.title("House price output and projections", size=24, pad=20)
  plt.tight_layout()
  display(plt.show())

def forecast_prix(model_name, model_stage):
  import pandas as pd
  from mlflow.tracking.client import MlflowClient
  client = MlflowClient()
  model_version = client.get_latest_versions(model_name, stages=[model_stage])[0].version
  model_uri = "models:/{model_name}/{model_stage}".format(model_name=model_name, model_stage=model_stage)
  model = mlflow.pyfunc.load_model(model_uri)
  # weather_data, past_power_output = get_weather_and_forecast()

  power_predictions = pd.DataFrame(model.predict(X_test))
  # power_predictions.index = pd.to_datetime(weather_data.index)
  #print(predictions)
  #print(power_predictions)
  plot(model_name, model_stage, int(model_version), power_predictions,predictions )

  


# COMMAND ----------

forecast_prix(model_name, "None")

# COMMAND ----------

client.transition_model_version_stage(
  name=model_name,
  version=latest_dev_version,
  stage='Staging',
)

# COMMAND ----------

#client.transition_model_version_stage(
#  name=model_name,
#  version=2,https://adb-2058505122667366.6.azuredatabricks.net/?o=2058505122667366#
#  stage="Archived"
#)


