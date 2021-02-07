import airflow_utils

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator


dag = airflow_utils.generate_dag(id='short_puts')
kwargs = airflow_utils.get_dag_kwargs(dag=dag)
dbt_kwargs = airflow_utils.get_dag_kwargs(dag=dag, type='dbt')
prediction_kwargs = airflow_utils.get_dag_kwargs(dag=dag, type='prediction')

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
