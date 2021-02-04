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
    dag_id='speculative_options',
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

dbt_train = DockerOperator(
    task_id='update_dbt_training_tables',
    command='dbt run -m speculation --profiles-dir .',
    **dbt_kwargs,
)

high_price_predictor = DockerOperator(
    task_id='high_price_predictor',
    command='python science/executor.py --job=s3 -d=1 --archive_files',
    **prediction_kwargs,
)

low_price_predictor = DockerOperator(
    task_id='low_price_predictor',
    command='python science/executor.py --job=s4 -d=0 --archive_files',
    **prediction_kwargs,
)

load_stock_predictions = DockerOperator(
    task_id='stock_prediction_loader',
    command='python data/science/predictions/loader.py',
    **kwargs,
)

dbt_trade = DockerOperator(
    task_id='update_dbt_trading_tables',
    command='dbt run -m speculative_options --profiles-dir .',
    **dbt_kwargs,
)

report_speculative_options = DockerOperator(
    task_id='speculative_options_report',
    command='python data/science/reports/speculative_options.py',
    **kwargs,
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

dbt_train.set_upstream(start_time)
high_price_predictor.set_upstream(dbt_train)
low_price_predictor.set_upstream(high_price_predictor)
load_stock_predictions.set_upstream(low_price_predictor)
dbt_trade.set_upstream(load_stock_predictions)
report_speculative_options.set_upstream(dbt_trade)
end_time.set_upstream(report_speculative_options)
