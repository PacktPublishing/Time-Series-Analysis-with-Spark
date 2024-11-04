from airflow import DAG
from airflow.operators.python_operator import PythonOperator

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta import *

from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
from mlflow.models import infer_signature

from pyspark.ml.evaluation import RegressionEvaluator

DAG_NAME = 'ts-spark_ch9_data-ml-ops'

DATASOURCE = (
    "/data/ts-spark_ch1_ds2.csv"
)
FORECAST = (
    "/data/ts-spark_ch9_prophet-forecast.csv"
)
ARTIFACT_DIR = "model"
np.random.seed(20244)

model_name = "ts-spark_ch9_data-ml-ops_time_series_prophet"
model_version = "latest"

builder = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("ts-spark_ch9_data-ml-ops_time_series_prophet") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

mlflow.set_tracking_uri("http://mlflow-server:5000")

def get_config(_vars, **kwargs):
    print(f"dag_config: {_vars}")

    return _vars

def ingest_train_data(_vars, **kwargs):
    sdf = spark.read.csv(DATASOURCE, header=True, inferSchema=True)
    sdf = sdf.filter((F.col('date') >= F.lit(_vars['START_DATE'])) & (F.col('date') <= F.lit(_vars['TRAIN_END_DATE'])))
    data_ingest_count = sdf.count()
    sdf.write.format("delta").mode("overwrite").save(f"/data/delta/ts-spark_ch9_bronze_train_{_vars['runid']}")

    _vars['train_ingest_count'] = data_ingest_count
    return _vars

def transform_train_data(_vars, **kwargs):
    sdf = spark.read.format("delta").load(f"/data/delta/ts-spark_ch9_bronze_train_{_vars['runid']}")
    sdf = sdf.selectExpr("date as ds", "cast(daily_min_temperature as double) as y")
    sdf = sdf.dropna()

    data_transform_count = sdf.count()
    sdf.write.format("delta").mode("overwrite").save(f"/data/delta/ts-spark_ch9_silver_train_{_vars['runid']}")

    _vars['train_transform_count'] = data_transform_count
    return _vars

def train_and_log_model(_vars, **kwargs):
    sdf = spark.read.format("delta").load(f"/data/delta/ts-spark_ch9_silver_train_{_vars['runid']}")
    pdf = sdf.toPandas()

    mlflow.set_experiment('ts-spark_ch9_data-ml-ops_time_series_prophet_train')
    mlflow.start_run()
    mlflow.log_param("DAG_NAME", DAG_NAME)
    mlflow.log_param("TRAIN_START_DATE", _vars['START_DATE'])
    mlflow.log_param("TRAIN_END_DATE", _vars['TRAIN_END_DATE'])
    mlflow.log_metric('train_ingest_count', _vars['train_ingest_count'])
    mlflow.log_metric('train_transform_count', _vars['train_transform_count'])

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

    mlflow.prophet.log_model(model, artifact_path=ARTIFACT_DIR, signature=signature, registered_model_name=model_name,)
    mlflow.log_params(param)
    mlflow.log_metrics(cv_metrics)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_DIR)

    print(f"CV params: \n{json.dumps(param, indent=2)}")
    print(f"CV metrics: \n{json.dumps(cv_metrics, indent=2)}")
    print(f"Model URI: {model_uri}")

    mlflow.end_run()
    return _vars

def forecast(_vars, **kwargs):

     # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    _model = mlflow.prophet.load_model(model_uri)

    forecast = _model.predict(_model.make_future_dataframe(periods=365, include_history = False))
    sdf = spark.createDataFrame(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    sdf.write.format("delta").mode("overwrite").save(f"/data/delta/ts-spark_ch9_gold_forecast_{_vars['runid']}")

    print(f"forecast:\n${forecast.tail(30)}")

    mlflow.end_run()
    return _vars

def ingest_eval_data(_vars, **kwargs):
    sdf = spark.read.csv(DATASOURCE, header=True, inferSchema=True)
    sdf = sdf.filter((F.col('date') >= F.lit(_vars['START_DATE'])) & (F.col('date') <= F.lit(_vars['EVAL_END_DATE'])))
    data_ingest_count = sdf.count()
    sdf.write.format("delta").mode("overwrite").save(f"/data/delta/ts-spark_ch9_bronze_eval_{_vars['runid']}")

    _vars['eval_ingest_count'] = data_ingest_count   
    return _vars

def transform_eval_data(_vars, **kwargs):
    sdf = spark.read.format("delta").load(f"/data/delta/ts-spark_ch9_bronze_eval_{_vars['runid']}")
    sdf = sdf.selectExpr("date as ds", "cast(daily_min_temperature as double) as y")
    sdf = sdf.dropna()

    #pdf = sdf.select("date", "daily_min_temperature").toPandas()
    #pdf.columns = ["ds", "y"]
    #pdf["y"] = pd.to_numeric(pdf["y"], errors="coerce")
    ##pdf.drop(index=pdf.index[-2:], inplace=True)
    #pdf.dropna(inplace = True)
    #sdf = spark.createDataFrame(pdf)

    data_transform_count = sdf.count()
    sdf.write.format("delta").mode("overwrite").save(f"/data/delta/ts-spark_ch9_silver_eval_{_vars['runid']}")

    _vars['eval_transform_count'] = data_transform_count
    return _vars

def eval_forecast(_vars, **kwargs):
    sdf = spark.read.format("delta").load(f"/data/delta/ts-spark_ch9_silver_eval_{_vars['runid']}")
    sdf_forecast = spark.read.format("delta").load(f"/data/delta/ts-spark_ch9_gold_forecast_{_vars['runid']}")
    sdf_eval = sdf.join(sdf_forecast, 'ds', "inner")
    eval_forecast_count = sdf_eval.count()

    print(f"sdf_eval:\n${sdf_eval.tail(10)}")
    sdf_eval_count = sdf_eval.count()
    print(f"sdf_eval count:\n${sdf_eval_count}")

    sdf_eval = sdf_eval.dropna()
 
    sdf_eval_count = sdf_eval.count()
    print(f"sdf_eval count:\n${sdf_eval_count}")

    evaluator = RegressionEvaluator(labelCol='y', predictionCol='yhat', metricName='rmse')
    eval_rmse = evaluator.evaluate(sdf_eval)

    #eval_rmse = 0
    #if (pdf_eval.shape[0] > 0):
    #    eval_rmse = mean_squared_error(pdf_eval['y'], pdf_eval['yhat'], squared=False)
    
    _vars['eval_forecast_count'] = eval_forecast_count
    _vars['eval_rmse'] = eval_rmse

    mlflow.set_experiment('ts-spark_ch9_data-ml-ops_time_series_prophet_eval')
    mlflow.start_run()
    mlflow.log_param("DAG_NAME", DAG_NAME)
    mlflow.log_param("EVAL_START_DATE", _vars['START_DATE'])
    mlflow.log_param("EVAL_END_DATE", _vars['EVAL_END_DATE'])
    mlflow.log_metric('eval_ingest_count', _vars['eval_ingest_count'])
    mlflow.log_metric('eval_transform_count', _vars['eval_transform_count'])
    mlflow.log_metric('eval_forecast_count', _vars['eval_forecast_count'])
    mlflow.log_metric('eval_rmse', _vars['eval_rmse'])
    mlflow.end_run()
    return _vars

# Define DAG: Set default args and schedule interval
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
 #   'retries': 1,
 #   'retry_delay': timedelta(minutes=1),
}

dag = DAG(DAG_NAME,
          default_args=default_args,
          description='ts-spark_ch9 - Data/MLOps pipeline example - Time series forecasting with Prophet',
          schedule_interval=None)

# Define tasks
t0 = PythonOperator(
    task_id='get_config',
    python_callable=get_config,
    op_kwargs={'_vars': {
            'runid': "{{ dag_run.conf['runid'] }}",
            'START_DATE': "{{ dag_run.conf['START_DATE'] }}",
            'TRAIN_END_DATE': "{{ dag_run.conf['TRAIN_END_DATE'] }}",
            'EVAL_END_DATE': "{{ dag_run.conf['EVAL_END_DATE'] }}",
            },
        },
    provide_context=True,
    dag=dag,
)

t1 = PythonOperator(
    task_id='ingest_train_data',
    python_callable=ingest_train_data,
    op_kwargs={'_vars': t0.output},    
    dag=dag,
)

t2 = PythonOperator(
    task_id='transform_train_data',
    python_callable=transform_train_data,
    op_kwargs={'_vars': t1.output},
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='train_and_log_model',
    python_callable=train_and_log_model,
    op_kwargs={'_vars': t2.output},
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='forecast',
    python_callable=forecast,
    op_kwargs={'_vars': t3.output},
    provide_context=True,
    dag=dag,
)

t5 = PythonOperator(
    task_id='ingest_eval_data',
    python_callable=ingest_eval_data,
    op_kwargs={'_vars': t4.output},
    dag=dag,
)

t6 = PythonOperator(
    task_id='transform_eval_data',
    python_callable=transform_eval_data,
    op_kwargs={'_vars': t5.output},
    provide_context=True,
    dag=dag,
)

t7 = PythonOperator(
    task_id='eval_forecast',
    python_callable=eval_forecast,
    op_kwargs={'_vars': t6.output},
    provide_context=True,
    dag=dag,
)

# Task dependencies
t0 >> t1 >> t2 >> t3 >> t4 >> t5 >> t6 >> t7