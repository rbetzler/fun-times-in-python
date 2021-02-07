from datetime import datetime, timedelta
from airflow.models import DAG

DEFAULT_ARGS = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['rbetzler94@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def generate_dag(
    id: str,
    default_args: dict=DEFAULT_ARGS,
):
    dag = DAG(
        dag_id=id,
        default_args=default_args,
        start_date=datetime(2019, 10, 29),
        schedule_interval='0 10 * * 7',
        catchup=False,
    )
    return dag

def get_dag_kwargs(
    dag,
    type: str='std',
) -> dict:
    kwargs = {
        'image': 'py-dw-stocks',
        'auto_remove': True,
        'volumes': ['/media/nautilus/fun-times-in-python:/usr/src/app'],
        'network_mode': 'bridge',
        'dag': dag,
    }
    if type == 'dbt':
        kwargs['volumes'] = ['/media/nautilus/fun-times-in-python/dbt:/usr/src/app']
    elif type == 'prediction':
        kwargs['image'] = 'pytorch'
    return kwargs
