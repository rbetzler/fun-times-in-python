import abc
import pandas as pd

from science import predictor
from science.utilities import science_utils
from utilities import utils

SYMBOL = 'symbol'
TARGET = 'target'
DENORMALIZED_TARGET = 'denormalized_target'
PREDICTION = 'prediction'
DENORMALIZED_PREDICTION = 'denormalized_prediction'
NORMALIZATION_MIN = 'normalization_min'
NORMALIZATION_MAX = 'normalization_max'


class Predictor(predictor.Predictor, abc.ABC):
    """Predict the high stocks prices over the next 30 days"""

    @property
    def get_symbols(self) -> pd.DataFrame:
        """Generate sql for one hot encoding columns in query"""
        query = '''
            select distinct ticker as symbol
            from nasdaq.listed_stocks
            where ticker !~ '[\^.~]'
              and character_length(ticker) between 1 and 4
            order by 1
            '''
        df = utils.query_db(query=query)
        return df

    @property
    def query(self) -> str:
        query = f'''
            select *
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


class PredictorS1(Predictor):
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


class PredictorS2(Predictor):
    @property
    def model_id(self) -> str:
        return 's2'

    @property
    def model_kwargs(self) -> dict:
        kwargs = {
            'n_layers': 2,
            'n_epochs': 500,
            'hidden_shape': 500,
            'dropout': 0.1,
            'learning_rate': .0001,
            'seed': 44,
            'sequence_length': 20,
            'batch_size': 1000,
        }
        return kwargs
