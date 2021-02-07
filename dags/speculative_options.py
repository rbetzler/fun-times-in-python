import airflow_utils

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator


dag = airflow_utils.generate_dag(id='speculative_options')
kwargs = airflow_utils.get_dag_kwargs(dag=dag)
dbt_kwargs = airflow_utils.get_dag_kwargs(dag=dag, type='dbt')
prediction_kwargs = airflow_utils.get_dag_kwargs(dag=dag, type='prediction')

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
    command='python science/executor.py --job=s4 -d=1 --archive_files',
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
