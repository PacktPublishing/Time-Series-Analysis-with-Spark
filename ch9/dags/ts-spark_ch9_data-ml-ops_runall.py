from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from datetime import datetime, timedelta

DAG_NAME = 'ts-spark_ch9_data-ml-ops_runall'

conf_run1 = {
  'runid':          1,
  'START_DATE':     '1981-01-01',
  'TRAIN_END_DATE': '1985-12-31',
  'EVAL_END_DATE':  '1986-12-31',
}
conf_run2 = {
  'runid':          2,
  'START_DATE':     '1982-01-01',
  'TRAIN_END_DATE': '1986-12-31',
  'EVAL_END_DATE':  '1987-12-31',
}
conf_run3 = {
  'runid':          3,
  'START_DATE':     '1983-01-01',
  'TRAIN_END_DATE': '1987-12-31',
  'EVAL_END_DATE':  '1988-12-31',
}
conf_run4 = {
  'runid':          4,
  'START_DATE':     '1984-01-01',
  'TRAIN_END_DATE': '1988-12-31',
  'EVAL_END_DATE':  '1989-12-31',
}
conf_run5 = {
  'runid':          5,
  'START_DATE':     '1985-01-01',
  'TRAIN_END_DATE': '1989-12-31',
  'EVAL_END_DATE':  '1990-12-31',
}

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
        description='ts-spark_ch9 - Data/MLOps pipeline example - Time series forecasting with Prophet (run all)',
        schedule_interval=None)

# Define tasks
t1 = TriggerDagRunOperator(
  task_id="ts-spark_ch9_data-ml-ops_1",
  trigger_dag_id="ts-spark_ch9_data-ml-ops",
  conf=conf_run1,
  wait_for_completion=True,
  dag=dag,
)

t2 = TriggerDagRunOperator(
  task_id="ts-spark_ch9_data-ml-ops_2",
  trigger_dag_id="ts-spark_ch9_data-ml-ops",
  conf=conf_run2,
  wait_for_completion=True,
  dag=dag,
)

t3 = TriggerDagRunOperator(
  task_id="ts-spark_ch9_data-ml-ops_3",
  trigger_dag_id="ts-spark_ch9_data-ml-ops",
  conf=conf_run3,
  wait_for_completion=True,
  dag=dag,
)

t4 = TriggerDagRunOperator(
  task_id="ts-spark_ch9_data-ml-ops_4",
  trigger_dag_id="ts-spark_ch9_data-ml-ops",
  conf=conf_run4,
  wait_for_completion=True,
  dag=dag,
)

t5 = TriggerDagRunOperator(
  task_id="ts-spark_ch9_data-ml-ops_5",
  trigger_dag_id="ts-spark_ch9_data-ml-ops",
  conf=conf_run5,
  wait_for_completion=True,
  dag=dag,
)

# Task dependencies
#t1
t1 >> t2 >> t3 >> t4 >> t5
#t1 >> t2 >> [t3, t4] >> t5
#t1 >> [t2, t3, t4] >> t5