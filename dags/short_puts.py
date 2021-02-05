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
    dag_id='short_puts',
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
dbt_kwargs = kwargs.copy()
dbt_kwargs['volumes'] = ['/media/nautilus/fun-times-in-python/dbt:/usr/src/app']

prediction_kwargs = kwargs.copy()
prediction_kwargs['image'] = 'pytorch'

start_time = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag,
)

dbt_training_technicals = DockerOperator(
    task_id='update_dbt_training_table',
    command='dbt run -m thirty_day_low technicals --profiles-dir .',
    **dbt_kwargs,
)

predict_stocks = DockerOperator(
    task_id='stock_predictor',
    command='python science/executor.py --job=s2 --archive_files',
    **prediction_kwargs,
)

load_stock_predictions = DockerOperator(
    task_id='stock_prediction_loader',
    command='python data/science/predictions/loader.py',
    **kwargs,
)

dbt_trading = DockerOperator(
    task_id='update_dbt_trading_tables',
    command='dbt run -m short_puts --profiles-dir .',
    **dbt_kwargs,
)

report_short_puts = DockerOperator(
    task_id='short_puts_report',
    command='python data/science/reports/short_puts.py',
    **kwargs,
)

endtime = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

dbt_training_technicals.set_upstream(start_time)
predict_stocks.set_upstream(dbt_training_technicals)
load_stock_predictions.set_upstream(predict_stocks)
dbt_trading.set_upstream(load_stock_predictions)
report_short_puts.set_upstream(dbt_trading)
endtime.set_upstream(report_short_puts)
