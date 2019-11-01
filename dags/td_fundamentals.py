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
    'retry_delay': timedelta(minutes = 1),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG(
    dag_id='td_fundamentals',
    default_args=args,
    start_date=datetime(2019, 10, 29),
    schedule_interval='0 12 * * 5',
    catchup=False
)

start_time = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag)

scrape = DockerOperator(
    task_id='scrape_td_fundamentals',
    image='py-dw-stocks',
    auto_remove=True,
    command='python finance/ingestion/td_ameritrade/fundamentals/scrape.py',
    volumes=['/media/nautilus/fun-times-in-python:/usr/src/app'],
    network_mode='bridge',
    dag=dag
)

load = DockerOperator(
    task_id='load_td_fundamentals',
    image='py-dw-stocks',
    auto_remove=True,
    command='python finance/ingestion/td_ameritrade/fundamentals/load.py',
    volumes=['/media/nautilus/fun-times-in-python:/usr/src/app'],
    network_mode='bridge',
    dag=dag
    )

table_creator = DockerOperator(
    task_id='update_td_fundamentals_table',
    image='py-dw-stocks',
    auto_remove=True,
    command='python finance/ingestion/td_ameritrade/fundamentals/table_creator.py',
    volumes=['/media/nautilus/fun-times-in-python:/usr/src/app'],
    network_mode='bridge',
    dag=dag
    )

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag)

scrape.set_upstream(start_time)
load.set_upstream(scrape)
table_creator.set_upstream(load)
end_time.set_upstream(table_creator)
