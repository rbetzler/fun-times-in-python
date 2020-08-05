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
    dag_id='td',
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

scrape_options = DockerOperator(
    task_id='scrape_td_options',
    command='python finance/data/td_ameritrade/options/scrape.py',
    **kwargs,
)

load_options = DockerOperator(
    task_id='load_td_options',
    command='python finance/data/td_ameritrade/options/load.py',
    **kwargs,
)

table_creator_options = DockerOperator(
    task_id='update_td_options_table',
    command='python finance/data/td_ameritrade/options/sql.py',
    **kwargs,
)

scrape_quotes = DockerOperator(
    task_id='scrape_td_quotes',
    command='python finance/data/td_ameritrade/quotes/scrape.py',
    **kwargs,
)

load_quotes = DockerOperator(
    task_id='load_td_quotes',
    command='python finance/data/td_ameritrade/quotes/load.py',
    **kwargs,
)

table_creator_quotes = DockerOperator(
    task_id='update_td_quotes_table',
    command='python finance/data/td_ameritrade/quotes/sql.py',
    **kwargs,
)

report_options = DockerOperator(
    task_id='report_options',
    command='python finance/science/reports/options.py',
    **kwargs,
)

scrape_fundamentals = DockerOperator(
    task_id='scrape_td_fundamentals',
    command='python finance/data/td_ameritrade/fundamentals/scrape.py',
    **kwargs,
)

load_fundamentals = DockerOperator(
    task_id='load_td_fundamentals',
    command='python finance/data/td_ameritrade/fundamentals/load.py',
    **kwargs,
)

table_creator_fundamentals = DockerOperator(
    task_id='update_td_fundamentals_table',
    command='python finance/data/td_ameritrade/fundamentals/sql.py',
    **kwargs,
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_options.set_upstream(start_time)
load_options.set_upstream(scrape_options)
table_creator_options.set_upstream(load_options)

scrape_quotes.set_upstream(scrape_options)
load_quotes.set_upstream(scrape_quotes)
table_creator_quotes.set_upstream(load_quotes)

report_options.set_upstream(table_creator_options)
report_options.set_upstream(table_creator_quotes)

scrape_fundamentals.set_upstream(scrape_quotes)
load_fundamentals.set_upstream(scrape_fundamentals)
table_creator_fundamentals.set_upstream(load_fundamentals)

end_time.set_upstream(report_options)
end_time.set_upstream(table_creator_fundamentals)
