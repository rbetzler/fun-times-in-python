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
    dag_id='science',
    default_args=args,
    start_date=datetime(2019, 10, 29),
    schedule_interval='0 10 * * 7',
    catchup=False,
)

kwargs = {
    'auto_remove': True,
    'image': 'py-dw-stocks',
    'volumes': ['/media/nautilus/fun-times-in-python:/usr/src/app'],
    'network_mode': 'bridge',
    'dag': dag,
}
prediction_kwargs = kwargs.copy()
prediction_kwargs['image'] = 'pytorch'

start_time = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag,
)

predict_stocks = DockerOperator(
    task_id='stock_predictor',
    command='python finance/science/executor.py --job=stock --archive_files=True',
    **prediction_kwargs,
)

load_stock_predictions = DockerOperator(
    task_id='stock_prediction_loader',
    command='python finance/data/science/predictions/loader.py',
    **kwargs,
)

decision_stocks = DockerOperator(
    task_id='stock_decision',
    command='python finance/science/dev/decisioner.py',
    **kwargs,
)

load_stock_decisions = DockerOperator(
    task_id='stock_decision_loader',
    command='python finance/data/science/decisions/loader.py',
    **kwargs,
)

report_stock_decisions = DockerOperator(
    task_id='stock_decision_report',
    command='python finance/data/science/decisions/report.py',
    **kwargs,
)

execute_trades = DockerOperator(
    task_id='execute_trades',
    command='python finance/trading/executor.py',
    **kwargs,
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

predict_stocks.set_upstream(start_time)
load_stock_predictions.set_upstream(predict_stocks)
decision_stocks.set_upstream(load_stock_predictions)
load_stock_decisions.set_upstream(decision_stocks)
report_stock_decisions.set_upstream(load_stock_decisions)
execute_trades.set_upstream(report_stock_decisions)
end_time.set_upstream(execute_trades)
