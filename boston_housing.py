# Databricks notebook source
pip install mlflow

# COMMAND ----------

pip install tensorflow

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
