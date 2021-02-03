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
dbt_kwargs = kwargs.copy()
dbt_kwargs['volumes'] = ['/media/nautilus/fun-times-in-python/dbt:/usr/src/app']

prediction_kwargs = kwargs.copy()
prediction_kwargs['image'] = 'pytorch'

start_time = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_options = DockerOperator(
    task_id='scrape_td_options',
    command='python data/td_ameritrade/options/scrape.py',
    **kwargs,
)

load_options = DockerOperator(
    task_id='load_td_options',
    command='python data/td_ameritrade/options/load.py',
    **kwargs,
)

dbt_options = DockerOperator(
    task_id='update_dbt_options_table',
    command='dbt run -m options --profiles-dir .',
    **dbt_kwargs,
)

scrape_quotes = DockerOperator(
    task_id='scrape_td_quotes',
    command='python data/td_ameritrade/quotes/scrape.py',
    **kwargs,
)

load_quotes = DockerOperator(
    task_id='load_td_quotes',
    command='python data/td_ameritrade/quotes/load.py',
    **kwargs,
)

dbt_quotes = DockerOperator(
    task_id='update_dbt_quotes_table',
    command='dbt run -m quotes --profiles-dir .',
    **dbt_kwargs,
)

dbt_stocks = DockerOperator(
    task_id='update_dbt_stocks_table',
    command='dbt run -m stocks --profiles-dir .',
    **dbt_kwargs,
)

report_options = DockerOperator(
    task_id='report_options',
    command='python data/td_ameritrade/options/report.py',
    **kwargs,
)

scrape_fundamentals = DockerOperator(
    task_id='scrape_td_fundamentals',
    command='python data/td_ameritrade/fundamentals/scrape.py',
    **kwargs,
)

load_fundamentals = DockerOperator(
    task_id='load_td_fundamentals',
    command='python data/td_ameritrade/fundamentals/load.py',
    **kwargs,
)

dbt_fundamentals = DockerOperator(
    task_id='update_dbt_fundamentals_table',
    command='dbt run -m fundamentals --profiles-dir .',
    **dbt_kwargs,
)

report_black_scholes = DockerOperator(
    task_id='report_black_scholes',
    command='python data/td_ameritrade/black_scholes/report.py',
    **kwargs,
)

load_black_scholes = DockerOperator(
    task_id='load_black_scholes',
    command='python data/td_ameritrade/black_scholes/load.py',
    **kwargs,
)

dbt_black_scholes = DockerOperator(
    task_id='update_dbt_black_scholes_table',
    command='dbt run -m black_scholes --profiles-dir .',
    **dbt_kwargs,
)

dbt_tests = DockerOperator(
    task_id='test_dbt_tables',
    command='dbt test --profiles-dir .',
    **dbt_kwargs,
)

dbt_training_technicals = DockerOperator(
    task_id='update_dbt_training_table',
    command='dbt run -m training technicals --profiles-dir .',
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
    command='dbt run -m short_puts speculative_options --profiles-dir .',
    **dbt_kwargs,
)

report_short_puts = DockerOperator(
    task_id='short_puts_report',
    command='python data/science/reports/short_puts.py',
    **kwargs,
)

report_speculative_options = DockerOperator(
    task_id='speculative_options_report',
    command='python data/science/reports/speculative_options.py',
    **kwargs,
)

execute_trades = DockerOperator(
    task_id='execute_trades',
    command='python trading/executor.py',
    **kwargs,
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_options.set_upstream(start_time)
load_options.set_upstream(scrape_options)
dbt_options.set_upstream(load_options)

scrape_quotes.set_upstream(scrape_options)
load_quotes.set_upstream(scrape_quotes)
dbt_quotes.set_upstream(load_quotes)
dbt_stocks.set_upstream(dbt_quotes)

report_black_scholes.set_upstream(dbt_options)
report_black_scholes.set_upstream(dbt_stocks)
load_black_scholes.set_upstream(report_black_scholes)
dbt_black_scholes.set_upstream(load_black_scholes)

report_options.set_upstream(dbt_options)
report_options.set_upstream(dbt_stocks)

scrape_fundamentals.set_upstream(scrape_quotes)
load_fundamentals.set_upstream(scrape_fundamentals)
dbt_fundamentals.set_upstream(load_fundamentals)

dbt_tests.set_upstream(dbt_black_scholes)
dbt_tests.set_upstream(dbt_fundamentals)

dbt_training_technicals.set_upstream(dbt_tests)
predict_stocks.set_upstream(dbt_training_technicals)
load_stock_predictions.set_upstream(predict_stocks)
dbt_trading.set_upstream(load_stock_predictions)
report_short_puts.set_upstream(dbt_trading)
report_speculative_options.set_upstream(dbt_trading)
execute_trades.set_upstream(dbt_trading)

end_time.set_upstream(execute_trades)
