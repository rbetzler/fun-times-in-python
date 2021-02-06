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
    dag_id='yahoo',
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
    task_id='scrape_yahoo_options',
    command='python data/yahoo/options/scrape.py',
    **kwargs,
)

scrape_income_statements = DockerOperator(
    task_id='scrape_yahoo_income_statements',
    command='python data/yahoo/income_statements/scrape.py',
    **kwargs,
)

scrape_balance_sheet = DockerOperator(
    task_id='scrape_yahoo_balance_sheet',
    command='python data/yahoo/balance_sheet/scrape.py',
    **kwargs,
)

scrape_cash_flow = DockerOperator(
    task_id='scrape_yahoo_cash_flow',
    command='python data/yahoo/cash_flow/scrape.py',
    **kwargs,
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_options.set_upstream(start_time)
scrape_income_statements.set_upstream(scrape_options)
scrape_balance_sheet.set_upstream(scrape_income_statements)
scrape_cash_flow.set_upstream(scrape_balance_sheet)
end_time.set_upstream(scrape_cash_flow)
