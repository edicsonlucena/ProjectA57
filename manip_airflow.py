# Databricks notebook source
!pip install --upgrade pip



# COMMAND ----------

!pip install "apache-airflow[databricks]"

# COMMAND ----------

!pip install "apache-airflow[databricks, celery, password]"

# COMMAND ----------

!pip install "apache-airflow[cncf.kubernetes]"

# COMMAND ----------

!airflow db init

# COMMAND ----------

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os
os.environ['AIRFLOW_HOME'] = '/content'

# Following are defaults  
default_args = {
    'owner': 'Edicson',
    'depends_on_past': False,
    'start_date': datetime(2021, 8, 8),
    'email': ['edicson.lucena@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('Airflow_Base', default_args=default_args)

# t1, t2, t3   are examples of tasks created using operators

t1 = BashOperator(
    task_id='task_1',
    bash_command='echo "Hello World from Task 1"',
    dag=dag)

t2 = BashOperator(
    task_id='task_2',
    bash_command='echo "Hello World from Task 2"',
    dag=dag)

t3 = BashOperator(
    task_id='task_3',
    bash_command='echo "Hello World from Task 3"',
    dag=dag)

 

t2.set_upstream(t1)
t3.set_upstream(t1)
 
 

# COMMAND ----------

# print the list of active DAGs
!airflow dags list

# COMMAND ----------

# prints the list of tasks in the "AirFlow" DAG
!airflow tasks list AirflowBase



# COMMAND ----------

# prints the hierarchy of tasks in the "AirFlow_Base" DAG
!airflow tasks list AirflowBase --tree

# COMMAND ----------

!airflow test hello_world task_1

# COMMAND ----------

example_bash_operator

# COMMAND ----------

import airflow

from airflow import DAG
from airflow.contrib.operators.databricks_operator import DatabricksSubmitRunOperator

# COMMAND ----------

args = {
    'owner': 'airflow',
    'email': ['airflow@example.com'],
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(0)
}

# COMMAND ----------

dag = DAG(dag_id='example_databricks_operator', default_args=args, schedule_interval='@daily')

# COMMAND ----------

new_cluster = {
    'spark_version': '6.0.x-scala2.11',
    "node_type_id": "Standard_D3_v2",
    'num_workers': 8
}

# COMMAND ----------

notebook_task_params = {
    'new_cluster': new_cluster,
    'notebook_task': {
    'notebook_path': '/Users/airflow@example.com/PrepareData',
  },
}
# Example of using the JSON parameter to initialize the operator.
notebook_task = DatabricksSubmitRunOperator(
  task_id='notebookA57_task',
  dag=dag,
  json=notebook_task_params)

# COMMAND ----------

# Example of using the named parameters of DatabricksSubmitRunOperator to initialize the operator.
spark_jar_task = DatabricksSubmitRunOperator(
  task_id='spark_jar_task',
  dag=dag,
  new_cluster=new_cluster,
  spark_jar_task={
    'main_class_name': 'com.example.ProcessData'
  },
  libraries=[
    {
      'jar': 'dbfs:/lib/etl-0.1.jar'
    }
  ]
)

# COMMAND ----------

notebook_task.set_downstream(spark_jar_task)

# COMMAND ----------

! airflow list_dags 

# COMMAND ----------

notebook_task = DatabricksSubmitRunOperator(
    task_id='notebook_task',
    dag=dag,
    json=notebook_task_params)

spark_jar_task = DatabricksSubmitRunOperator(
    task_id='spark_jar_task',
    dag=dag,
    json=spark_jar_task_params)

notebook_task.set_downstream(spark_jar_task)


# COMMAND ----------

!airflow dags list

# COMMAND ----------

!airflow tasks list example_databricks_operator

# COMMAND ----------

!pip install --upgrade "git+git://github.com/databricks/incubator-airflow.git@1.8.1-db1#egg=apache-airflow[databricks]"

# COMMAND ----------

!airflow db init

# COMMAND ----------

!airflow webserver      

# COMMAND ----------


