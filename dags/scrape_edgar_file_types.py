

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


#This didn't run but whatever
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 1, 1)
}

container = 'py-temp'
script = '/home/py-scripts/web-scraping/scrapers/scrape_edgar_file_types.py'
templated_executor = "python /usr/local/airflow_home/utilities/airflow_container_executor.py " + container + " " + script

dag = DAG(
    'scrape_edgar_file_types',
    default_args = default_args,
    schedule_interval = timedelta(days = 1000))

t1 = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag)

t2 = BashOperator(
    task_id = 'scrape_edgar_file_types',
    bash_command = templated_executor,
    dag = dag)

t3 = BashOperator(
    task_id = 'end_pipeline',
    bash_command = 'date',
    dag = dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
