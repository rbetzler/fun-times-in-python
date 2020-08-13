"""modeling utils"""
import datetime
import pandas as pd


def save_file(
        df: pd.DataFrame,
        name: str,
        is_prod: bool = False,
        delimiter: str = ',',
):
    """Save dataframe to science folder"""
    if is_prod:
        subfolder = 'prod'
    else:
        subfolder = 'dev'

    path = f"/usr/src/app/audit/science/{subfolder}/{name}/{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
    df.to_csv(
        path,
        index=False,
        header=True,
        sep=delimiter,
    )
