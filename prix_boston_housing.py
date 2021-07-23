# Databricks notebook source
pip install mlflow

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

# Import the dataset from scikit-learn and create the training and test datasets. 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import mlflow.tensorflow


db = load_boston()

X = db.data
y = db.target
X_train, X_test, y_train, y_test = train_test_split(X, y)


# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# In this run, neither the experiment_id nor the experiment_name parameter is provided. MLflow automatically creates a notebook experiment and logs runs to it.
# Access these runs using the Experiment sidebar. Click Experiment at the upper right of this screen. 
with mlflow.start_run():
  n_estimators = 100
  max_depth = 6
  max_features = 3
  
    # Automatically capture the model's parameters, metrics, artifacts,
  # and source code with the `autolog()` function
  mlflow.sklearn.autolog()
  
  # Create and train model
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  run_id = mlflow.active_run().info.run_id
    
  # Make predictions
  predictions = rf.predict(X_test)
  
  

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# In this run, neither the experiment_id nor the experiment_name parameter is provided. MLflow automatically creates a notebook experiment and logs runs to it.
# Access these runs using the Experiment sidebar. Click Experiment at the upper right of this screen. 
with mlflow.start_run():
  n_estimators = 100
  max_depth = 6
  max_features = 3
  # Create and train model
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  # Make predictions
  predictions = rf.predict(X_test)
  
  # Log parameters
  mlflow.log_param("num_trees", n_estimators)
  mlflow.log_param("maxdepth", max_depth)
  mlflow.log_param("max_feat", max_features)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
    
  # Log metrics
  mlflow.log_metric("mse", mse)

# COMMAND ----------

# This run uses mlflow.set_experiment() to specify an experiment in the workspace where runs should be logged. 
# If the experiment specified by experiment_name does not exist in the workspace, MLflow creates it.
# Access these runs using the experiment name in the workspace file tree. 

experiment_name = "/Shared/boston_housing_experiment/"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
  n_estimators = 100
  max_depth = 6
  max_features = 3
  # Create and train model
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  run_id = mlflow.active_run().info.run_id
    
  # Make predictions
  predictions = rf.predict(X_test)
  
  # Log parameters
  mlflow.log_param("num_trees", n_estimators)
  mlflow.log_param("maxdepth", max_depth)
  mlflow.log_param("max_feat", max_features)
  
  # Log model
  mlflow.sklearn.log_model(rf, "random-forest-model")
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
    
  # Log metrics
  mlflow.log_metric("mse", mse)
  


# COMMAND ----------

print(predictions)

# COMMAND ----------

print (run_id)

# COMMAND ----------

# MAGIC %md #  Create a new registered model using the API

# COMMAND ----------

model_name = "boston-housing-model" # Replace this with the name of your registered model, if necessary.

# COMMAND ----------

import mlflow

# The default path where the MLflow autologging function stores the model
artifact_path = "model"
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md After creating a model version, it may take a short period of time to become ready. Certain operations, such as model stage transitions, require the model to be in the READY state. Other operations, such as adding a description or fetching model details, can be performed before the model version is ready (for example, while it is in the PENDING_REGISTRATION state).
# MAGIC 
# MAGIC The following cell uses the MlflowClient.get_model_version() function to wait until the model is ready.

# COMMAND ----------

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)
  
wait_until_ready(model_details.name, model_details.version)

# COMMAND ----------

# MAGIC %md ### Add model descriptions

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
client.update_registered_model(
  name=model_details.name,
  description= "Ce modèle prédit le prix de la maison"
)

# COMMAND ----------

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="Cette version du modèle a été construite à l'aide de sklearn. C'est un déploiement d'algorithme Random Forest Regressor"
)

# COMMAND ----------

# MAGIC %md  transition the model to the Production stage.

# COMMAND ----------

client.transition_model_version_stage(
  name=model_details.name,
  version=model_details.version,
  stage='Production',
)

# COMMAND ----------

model_version_details = client.get_model_version(
  name=model_details.name,
  version=model_details.version,
)
print("The current model stage is: '{stage}'".format(stage=model_version_details.current_stage))

# COMMAND ----------

latest_version_info = client.get_latest_versions(model_name, stages=["Production"])
latest_production_version = latest_version_info[0].version
print("The latest production version of the model '%s' is '%s'." % (model_name, latest_production_version))


# COMMAND ----------

import mlflow.pyfunc

model_version_uri = "models:/{model_name}/1".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
model_version_1 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

model_production_uri = "models:/{model_name}/production".format(model_name=model_name)

print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
model_production = mlflow.pyfunc.load_model(model_production_uri)

# COMMAND ----------

# MAGIC %md ## Prediction prix output with the production model

# COMMAND ----------

X

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

forecast_prix(model_name, "Production")

# COMMAND ----------


