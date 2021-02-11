import abc
import pandas as pd

from science.models import autoencoder
from science.predictor import base

MARKET_DATETIME = 'market_datetime'
SYMBOL = 'symbol'


class StockAutoencoder(base.Predictor):
    """Subclass for stock autoencoder"""

    @property
    def output_folder(self) -> str:
        return 'pred_autoencoder'

    @property
    def columns_to_ignore(self) -> list:
        cols = [
            MARKET_DATETIME,
            SYMBOL,
        ]
        return cols

    @property
    def query(self) -> str:
        query = f'''
            select
                market_datetime
              , symbol
              , open_1
              , open_2
              , open_3
              , open_4
              , open_5
              , open_6
              , open_7
              , open_8
              , open_9
              , open_10
              , open_11
              , open_12
              , open_13
              , open_14
              , open_15
              , open_16
              , open_17
              , open_18
              , open_19
              , open_20
              , open_21
              , open_22
              , open_23
              , open_24
              , open_25
              , open_26
              , open_27
              , open_28
              , open_29
            from dbt_train.thirty_day_low
            where market_datetime between '{self.start_date}' and '{self.end_date}'
            order by market_datetime, symbol
            { 'limit ' + str(self.limit) if self.is_training_run else '' }
            { 'offset ' + str(self.limit * self.n_subrun) if self.is_training_run else '' }
            '''
        return query

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        output['model_id'] = self.model_id
        df = pd.merge(
            input,
            output,
            left_index=True,
            right_index=True,
            suffixes=('_target', '_prediction'),
        )
        return df

    @property
    def target_column(self) -> str:
        pass

    def model(self, df: pd.DataFrame):
        model = autoencoder.NNAutoencoder0(
            x=df.drop(self.columns_to_ignore, axis=1),
            y=df.drop(self.columns_to_ignore, axis=1),
            output_shape=len(df.drop(self.columns_to_ignore, axis=1).columns),
            device=self.device,
            **self.model_kwargs,
        )
        return model
