import pandas as pd
from finance.data import loader

COLUMNS = [
    'model_id',
    'market_datetime',
    'symbol',
    'target',
    'denormalized_target',
    'prediction',
    'denormalized_prediction',
    'normalization_min',
    'normalization_max',
    'file_datetime',
]


class DevTradesLoader(loader.FileIngestion):

    @property
    def job_name(self) -> str:
        return 'dev_trades_loader'

    @property
    def directory(self) -> str:
        return 'science/dev/trades'

    @property
    def import_file_prefix(self) -> str:
        return 'v'

    @property
    def schema(self) -> str:
        return 'dev'

    @property
    def table(self) -> str:
        return 'trades'

    def clean_df(self, df) -> pd.DataFrame:
        df = df[COLUMNS]
        return df


if __name__ == '__main__':
    DevTradesLoader().execute()
