from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append("/opt/airflow/mlops_code")

from mlflow.tasks.data_preparation import prepare_data
from mlflow.tasks.hyperparameter_tuning import hyperparameter_tuning
from mlflow.tasks.evaluate import evaluate

with DAG("knn_pipeline", start_date=datetime(2024, 1, 1), schedule_interval=None, catchup=False) as dag:

    prep_task = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    tune_task = PythonOperator(task_id="tune_model", python_callable=hyperparameter_tuning)
    eval_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate)

    prep_task >> tune_task >> eval_task
