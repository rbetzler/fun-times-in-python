from __future__ import print_function

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator
from datetime import datetime, timedelta


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
    dag_id='table_creator',
    default_args=args,
    schedule_interval=None,
)

start_time = BashOperator(
    task_id = 'start_pipeline',
    bash_command = 'date',
    dag = dag)

end_time = BashOperator(
    task_id = 'end_pipeline',
    bash_command = 'date',
    dag = dag)

tasks = {}
command_prefix = 'python finance/ingestion/'
command_suffix = '/table_creator.py'
jobs = ['edgar', 'fred', 'internals', 'td_ameritrade', 'yahoo']
for job in jobs:
    tasks.update({job: command_prefix + job + command_suffix})

prior_task = ''
for task in tasks:
    task_id = 'create_tables_' + task
    dock_task = DockerOperator(
        task_id = task_id,
        image = 'py-dw-stocks',
        auto_remove = True,
        command = tasks.get(task),
        volumes = ['/media/nautilus/fun-times-in-python:/usr/src/app'],
        network_mode = 'bridge',
        dag = dag
        )
    if prior_task:
        dock_task.set_upstream(prior_task)
    else: dock_task.set_upstream(start_time)
    prior_task = dock_task

end_time.set_upstream(dock_task)