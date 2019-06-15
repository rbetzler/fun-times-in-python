
from __future__ import print_function

import time
from builtins import range
from pprint import pprint

import airflow
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

import sys
import docker

args = {
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

dag = DAG(
    dag_id='operator_python_dockerized',
    default_args=args,
    schedule_interval=None,
)

def run_python_in_docker(docker_container, airflow_script):
    client = docker.from_env()
    client.start(docker_container)
    py_client = client.exec_create(container = docker_container, cmd = 'python ' + airflow_script)
    client.exec_start(py_client['Id'])
    client.stop(docker_container)

task = PythonOperator(
    task_id = 'test_dockerized_',
    python_callable = run_python_in_docker,
    op_kwargs = {
        'docker_container' : 'py-temp',
        'airflow_script' : '/home/py-scripts/web-scraping/yahoo/execute_yahoo.py'},
    email_on_failure = True,
    email = 'rbetzler94@gmail.com',
    dag = dag
)

#works but error handling is not good
