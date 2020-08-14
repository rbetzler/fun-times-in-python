"""modeling utils"""
import datetime
import pandas as pd


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
