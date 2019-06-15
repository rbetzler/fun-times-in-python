

from __future__ import print_function

import time
from builtins import range
from pprint import pprint

import airflow
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.docker_operator import DockerOperator
from datetime import datetime, timedelta

import sys
import docker

args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'email': ['rbetzler94@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes = 1),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG(
    dag_id='yahoo_stocks',
    default_args=args,
    schedule_interval=None,
)

start_time = BashOperator(
    task_id = 'start_pipeline',
    bash_command = 'date',
    dag = dag)

task = DockerOperator(
    task_id = 'scrape_yahoo_stocks',
    image = 'py-dw-stocks',
    auto_remove = True,
    command = 'python /home/py-scripts/web-scraping/yahoo/execute_yahoo.py',
    volumes = ['/home/nautilus/development/fun-times-in-python:/home', '/media/nautilus/raw-files:/mnt'],
    network_mode = 'local-network',
    dag = dag
    )

end_time = BashOperator(
    task_id = 'end_pipeline',
    bash_command = 'date',
    dag = dag)

task.set_upstream(start_time)
end_time.set_upstream(task)
