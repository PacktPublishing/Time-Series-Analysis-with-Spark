from airflow import DAG
from airflow.operators.python_operator import PythonOperator

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
from mlflow.models import infer_signature

DATASOURCE = (
    "/data/ts-spark_ch1_ds2.csv"
)
ARTIFACT_DIR = "model"
np.random.seed(20244)

spark = SparkSession.builder \
        .master("spark://spark-master:7077") \
        .appName("ts-spark_ch4_data-ml-ops_time_series_prophet") \
        .getOrCreate()

mlflow.set_tracking_uri("http://mlflow-server:5000")
 
def ingest_data():
    sdf = spark.read.csv(DATASOURCE, header=True, inferSchema=True)
    pdf = sdf.select("date", "daily_min_temperature").toPandas()
    return pdf

def transform_data(pdf, **kwargs):
    pdf.columns = ["ds", "y"]
    pdf["y"] = pd.to_numeric(pdf["y"], errors="coerce")
    pdf.drop(index=pdf.index[-2:], inplace=True)
    pdf.dropna()
    return pdf

def train_and_log_model(pdf, **kwargs):
    mlflow.set_experiment('ts-spark_ch4_data-ml-ops_time_series_prophet')

    with mlflow.start_run():    
        model = Prophet().fit(pdf)

        param = {attr: getattr(model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}

        cv_metrics_name = ["mse", "rmse", "mae", "mdape", "smape", "coverage"]
        cv_params = cross_validation(
            model=model,
            horizon="90 days",
            period="30 days",
            initial="700 days",
            parallel="threads",
            disable_tqdm=True,
        )
        _cv_metrics = performance_metrics(cv_params)
        cv_metrics = {n: _cv_metrics[n].mean() for n in cv_metrics_name}

        train = model.history
        predictions = model.predict(model.make_future_dataframe(30))
        signature = infer_signature(train, predictions)

        mlflow.prophet.log_model(model, artifact_path=ARTIFACT_DIR, signature=signature)
        mlflow.log_params(param)
        mlflow.log_metrics(cv_metrics)
        model_uri = mlflow.get_artifact_uri(ARTIFACT_DIR)

        print(f"CV params: \n{json.dumps(param, indent=2)}")
        print(f"CV metrics: \n{json.dumps(cv_metrics, indent=2)}")
        print(f"Model URI: {model_uri}")

        mlflow.end_run()
        return model_uri

def forecast(model_uri, **kwargs):
    _model = mlflow.prophet.load_model(model_uri)

    forecast = _model.predict(_model.make_future_dataframe(30))
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('/data/ts-spark_ch4_prophet-forecast.csv')

    print(f"forecast:\n${forecast.head(10)}")

    return '/data/ts-spark_ch4_prophet-forecast.csv'

# Define DAG: Set default args and schedule interval
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG('ts-spark_ch4_data-ml-ops_time_series_prophet',
          default_args=default_args,
          description='ts-spark_ch4 - Data/MLOps pipeline example - Time series forecasting with Prophet',
          schedule_interval=None)

# Define tasks
t1 = PythonOperator(
    task_id='ingest_data',
    python_callable=ingest_data,
    dag=dag,
)

t2 = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    op_kwargs={'pdf': t1.output},
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='train_and_log_model',
    python_callable=train_and_log_model,
    op_kwargs={'pdf': t2.output},
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='forecast',
    python_callable=forecast,
    op_kwargs={'model_uri': t3.output},
    provide_context=True,
    dag=dag,
)

# Task dependencies
t1 >> t2 >> t3 >> t4
