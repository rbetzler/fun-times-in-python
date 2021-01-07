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
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    dag_id='td_equities',
    default_args=args,
    start_date=datetime(2019, 10, 29),
    schedule_interval='0 10 * * 7',
    catchup=False,
)

kwargs = {
    'image': 'py-dw-stocks',
    'auto_remove': True,
    'volumes': ['/media/nautilus/fun-times-in-python:/usr/src/app'],
    'network_mode': 'bridge',
    'dag': dag,
}

start_time = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_equities = DockerOperator(
    task_id='scrape_td_equities',
    command='python data/td_ameritrade/equities/scrape.py',
    **kwargs,
)

load_equities = DockerOperator(
    task_id='load_td_equities',
    command='python data/td_ameritrade/equities/load.py',
    **kwargs,
)

table_creator_equities = DockerOperator(
    task_id='update_td_equities_table',
    command='python data/td_ameritrade/equities/sql.py',
    **kwargs,
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_equities.set_upstream(start_time)
load_equities.set_upstream(scrape_equities)
table_creator_equities.set_upstream(load_equities)
end_time.set_upstream(table_creator_equities)
