"""
Code that goes along with the Airflow tutorial located at:
https://github.com/apache/airflow/blob/master/airflow/example_dags/tutorial.py
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 1, 1),
    'schedule_interval': 'None'
}

container = 'py-temp'
script = '/home/py-scripts/sql-execution/create_dw_stocks.py'
templated_executor = "python /usr/local/airflow_home/utilities/airflow_container_executor.py " + container + " " + script

dag = DAG(
    'build_dw_stocks',
    default_args = default_args,
    schedule_interval = timedelta(minutes = 10))

t1 = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag)

t2 = BashOperator(
    task_id = 'build_dw_stocks',
    bash_command = templated_executor,
    dag = dag)

t3 = BashOperator(
    task_id = 'end_pipeline',
    bash_command = 'date',
    dag = dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
