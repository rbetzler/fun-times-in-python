import abc
import pandas as pd

from science.predictor import base
from science.utilities import science_utils
from utilities import utils

SYMBOL = 'symbol'
TARGET = 'target'
DENORMALIZED_TARGET = 'denormalized_target'
PREDICTION = 'prediction'
DENORMALIZED_PREDICTION = 'denormalized_prediction'
NORMALIZATION_MIN = 'normalization_min'
NORMALIZATION_MAX = 'normalization_max'


class Predictor(base.Predictor, abc.ABC):
    """Predict the high stocks prices over the next 30 days"""

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

    @property
    def query(self) -> str:
        query = f'''
            select
                symbol
              , market_datetime
              , target
              , denormalized_target
              , normalization_min
              , normalization_max
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
              , open_30
            from dbt.training
            where market_datetime between '{self.start_date}' and '{self.end_date}'
              {'and target is not null' if self.is_training_run else ''}
            order by market_datetime, symbol
            '''
        return query

    @property
    def target_column(self) -> str:
        return TARGET

    @property
    def columns_to_ignore(self) -> list:
        cols = [
            'market_datetime',
            SYMBOL,
            DENORMALIZED_TARGET,
            NORMALIZATION_MIN,
            NORMALIZATION_MAX,
        ] + [self.target_column]
        return cols

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data pre-model run"""
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
        df[DENORMALIZED_PREDICTION] = df[PREDICTION] * (df[NORMALIZATION_MAX] - df[NORMALIZATION_MIN]) + df[NORMALIZATION_MIN]
        return df


class S1(Predictor):
    @property
    def model_id(self) -> str:
        return 's1'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 500,
            'hidden_shape': 1000,
            'dropout': 0.1,
            'learning_rate': .0001,
            'seed': 44,
            'sequence_length': 20,
            'batch_size': 13000,
        }
        return kwargs
