"""modeling utils"""
import datetime
import pandas as pd
from utilities import utils


def save_file(
        df: pd.DataFrame,
        subfolder: str,
        filename: str = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S'),
        is_prod: bool = False,
        delimiter: str = ',',
):
    """Save dataframe to science folder"""

    folder = 'prod' if is_prod else 'dev'

    path = f'/usr/src/app/audit/science/{folder}/{subfolder}/{filename}.csv'
    df.to_csv(
        path,
        index=False,
        header=True,
        sep=delimiter,
    )


def get_latest_market_datetime():
    query = 'select max(market_datetime)::date as market_datetime from dbt.stocks;'
    df = utils.query_db(query=query)
    latest_market_datetime = df['market_datetime'].values[0].strftime('%Y-%m-%d')
    return latest_market_datetime
