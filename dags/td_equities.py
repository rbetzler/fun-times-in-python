import airflow_utils

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.docker_operator import DockerOperator


dag = airflow_utils.generate_dag(id='td_equities')
kwargs = airflow_utils.get_dag_kwargs(dag=dag)
dbt_kwargs = airflow_utils.get_dag_kwargs(dag=dag, type='dbt')

start_time = BashOperator(
    task_id='start_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_equities = DockerOperator(
    task_id='scrape_td_equities',
    command='python data/td_ameritrade/equities/scrape.py',
    **kwargs,
)

load_equities = DockerOperator(
    task_id='load_td_equities',
    command='python data/td_ameritrade/equities/load.py',
    **kwargs,
)

table_creator_equities = DockerOperator(
    task_id='update_td_equities_table',
    command='dbt run -m equities stocks --profiles-dir .',
    **dbt_kwargs,
)

end_time = BashOperator(
    task_id='end_pipeline',
    bash_command='date',
    dag=dag,
)

scrape_equities.set_upstream(start_time)
load_equities.set_upstream(scrape_equities)
table_creator_equities.set_upstream(load_equities)
end_time.set_upstream(table_creator_equities)
