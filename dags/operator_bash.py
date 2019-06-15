

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

container = 'py-temp'
script = '/home/py-scripts/utilities/test_script.py'
templated_executor = "python /usr/local/airflow_home/utilities/airflow_container_executor.py " + container + " " + script

dag = DAG(
    'pipeline_template',
    default_args = default_args,
    schedule_interval = timedelta(minutes = 10))

t1 = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag)

t2 = BashOperator(
    task_id = 'execute_container',
    bash_command = templated_executor,
    dag = dag)

t3 = BashOperator(
    task_id = 'end_pipeline',
    bash_command = 'date',
    dag = dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
