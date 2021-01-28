import abc
import datetime
import os
import pandas as pd
import torch

from science import core
from science.utilities import modeling_utils, lstm_utils, science_utils
from utilities import utils

SYMBOL = 'symbol'
TARGET = 'target'
DENORMALIZED_TARGET = 'denormalized_target'
PREDICTION = 'prediction'
DENORMALIZED_PREDICTION = 'denormalized_prediction'
NORMALIZATION_MIN = 'normalization_min'
NORMALIZATION_MAX = 'normalization_max'


class Predictor(core.Science, abc.ABC):
    def __init__(
            self,
            n_days: int = 1000,
            is_training_run: bool = False,
            n_subrun: int = None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.end_date = self.start_date + datetime.timedelta(days=int(n_days))
        self._is_training_run = is_training_run
        self.n_subrun = n_subrun

    @property
    def is_training_run(self) -> bool:
        """Whether the model will be trained or not"""
        return self._is_training_run

    @property
    def n_subruns(self) -> int:
        """When backtesting, the number of iterations for a given date range"""
        return 2

    @property
    def limit(self) -> int:
        """When backtesting, the size of the dataset"""
        return 62000

    @property
    def trained_model_filepath(self) -> str:
        """Filepath from where to load and to where to save a trained model"""
        return f'/usr/src/app/audit/science/{self.location}/models/{self.model_id}'

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
        """Process data pre-model run"""
        symbols = self.get_symbols
        final = science_utils.encode_one_hot(
            df=df,
            column='symbol',
            keys=symbols['symbol'].to_list(),
        )
        return final

    @property
    @abc.abstractmethod
    def model_kwargs(self) -> dict:
        """LSTM model keyword arguments"""
        pass

    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        output['model_id'] = self.model_id
        df = input[self.columns_to_ignore].join(output)
        df[DENORMALIZED_PREDICTION] = df[PREDICTION] * (df[NORMALIZATION_MAX] - df[NORMALIZATION_MIN]) + df[NORMALIZATION_MIN]
        return df

    def execute(self):
        print(f'''
        Timestamp: {datetime.datetime.utcnow()}
        Model ID: {self.model_id}
        Location: {self.location}
        Training: {self.is_training_run}
        Archive: {self.archive_files}
        Start Date: {self.start_date}
        End Date: {self.end_date}
        Subrun: {self.n_subrun}
        ''')

        print(f'Getting raw data {datetime.datetime.utcnow()}')
        df = utils.query_db(query=self.query)

        # TODO: Raise an error if data is missing; accommodate holidays
        if not df.empty:

            print(f'Pre-processing raw data {datetime.datetime.utcnow()}')
            input = self.preprocess_data(df)

            print(f'Configuring model {datetime.datetime.utcnow()}')
            model = lstm_utils.TorchLSTM(
                x=input.drop(self.columns_to_ignore, axis=1),
                y=input[self.target_column],
                **self.model_kwargs,
            )

            if os.path.isfile(self.trained_model_filepath):
                print(f'Loading pre-trained model {datetime.datetime.utcnow()}')
                trained_model_params = torch.load(self.trained_model_filepath)
                model.load_state_dict(trained_model_params)

            if self.is_training_run:
                print(f'Fitting model {datetime.datetime.utcnow()}')
                model.fit()

                print(f'Saving model to {self.trained_model_filepath}: {datetime.datetime.utcnow()}')
                torch.save(model.state_dict(), self.trained_model_filepath)

            print(f'Generating prediction {datetime.datetime.utcnow()}')
            output = model.prediction_df

            print(f'Post-processing data {datetime.datetime.utcnow()}')
            predictions = self.postprocess_data(input=input, output=output)
            if self.archive_files:
                print(f'Saving model predictions to {self.location} {datetime.datetime.utcnow()}')
                modeling_utils.save_file(
                    df=predictions,
                    subfolder='predictions',
                    filename=self.filename(self.model_id),
                    is_prod=self.is_prod,
                )

        else:
            print(f'Dataframe is empty. Check if data is missing: {self.start_date}')
