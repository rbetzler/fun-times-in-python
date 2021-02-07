import airflow_utils

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator


dag = airflow_utils.generate_dag(id='yahoo')
kwargs = airflow_utils.get_dag_kwargs(dag=dag)
dbt_kwargs = airflow_utils.get_dag_kwargs(dag=dag, type='dbt')

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

scrape_sp = DockerOperator(
    task_id='scrape_yahoo_sp',
    command='python data/yahoo/sp/scrape.py',
    **kwargs,
)

load_sp = DockerOperator(
    task_id='load_yahoo_sp',
    command='python data/yahoo/sp/load.py',
    **kwargs,
)

dbt_sp = DockerOperator(
    task_id='update_dbt_sp',
    command='dbt run -m sp --profiles-dir .',
    **dbt_kwargs,
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

scrape_sp.set_upstream(scrape_cash_flow)
load_sp.set_upstream(scrape_sp)
dbt_sp.set_upstream(load_sp)
end_time.set_upstream(dbt_sp)
