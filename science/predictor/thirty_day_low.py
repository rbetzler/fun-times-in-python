import abc
import pandas as pd

from science.predictor import base
from utilities import science_utils, utils

MARKET_DATETIME = 'market_datetime'
OPEN = 'open'
PREDICTION = 'prediction'
SCALED_PREDICTION = 'scaled_prediction'
SCALED_TARGET = 'scaled_target'
SYMBOL = 'symbol'


class ThirtyDayLowPredictor(base.Predictor, abc.ABC):
    """Predict the lowest stock prices over the next 30 days"""

    @property
    def columns_to_ignore(self) -> list:
        cols = [
            SYMBOL,
            MARKET_DATETIME,
            SCALED_TARGET,
            OPEN,
        ] + [self.target_column]
        return cols

    @property
    def query(self) -> str:
        query = f'''
            select
                symbol
              , market_datetime
              , scaled_target
              , target
              , open
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
            from dbt.training
            where market_datetime between '{self.start_date}' and '{self.end_date}'
              {'and target is not null' if self.is_training_run else ''}
            order by market_datetime, symbol
            { 'limit ' + str(self.limit) if self.is_training_run else '' }
            { 'offset ' + str(self.limit * self.n_subrun) if self.is_training_run else '' }
            '''
        return query

    @property
    def get_symbols(self) -> pd.DataFrame:
        """Generate sql for one hot encoding columns in query"""
        query = '''
            select symbol
            from dbt.tickers
            order by 1
            '''
        df = utils.query_db(query=query)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        symbols = self.get_symbols
        final = science_utils.encode_one_hot(
            df=df,
            column='symbol',
            keys=symbols['symbol'].to_list(),
        )
        return final

    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        output['model_id'] = self.model_id
        df = input[self.columns_to_ignore].join(output)
        df[SCALED_PREDICTION] = (1 - df[PREDICTION]) * df[OPEN]
        return df
