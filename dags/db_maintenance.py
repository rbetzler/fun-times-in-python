from __future__ import print_function

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator
from datetime import datetime, timedelta


args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['rbetzler94@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=30)
}

dag = DAG(
    dag_id='db_maintenance',
    default_args=args,
    start_date=datetime(2019, 10, 29),
    schedule_interval='0 10 * * 6',
    catchup=False
)

start_time = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag)

maintenance = DockerOperator(
    task_id='db_maintenance',
    image='py-dw-stocks',
    auto_remove=True,
    command='python finance/data/internals/maintenance.py',
    volumes=['/media/nautilus/fun-times-in-python:/usr/src/app'],
    network_mode='bridge',
    dag=dag
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag)

maintenance.set_upstream(start_time)
end_time.set_upstream(maintenance)
