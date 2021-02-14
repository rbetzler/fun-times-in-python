import abc
import datetime
import os
import pandas as pd
import torch

from science import core
from utilities import utils, modeling_utils


class Predictor(core.Science, abc.ABC):
    def __init__(
            self,
            n_days: int = 1000,
            is_training_run: bool = False,
            n_subrun: int = None,
            device: int = 0,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.end_date = self.start_date + datetime.timedelta(days=int(n_days))
        self._is_training_run = is_training_run
        self.n_subrun = n_subrun
        self.device = device

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
    def output_subfolder(self) -> str:
        """Filepath from where to load and to where to save a trained model"""
        return 'predictions'

    @property
    @abc.abstractmethod
    def model_kwargs(self) -> dict:
        """LSTM model keyword arguments"""
        pass

    @abc.abstractmethod
    def model(self):
        """Pytorch nn to implement"""
        pass

    @property
    def target_column(self) -> str:
        """Column for model to predict"""
        return 'target'

    @property
    @abc.abstractmethod
    def columns_to_ignore(self) -> list:
        """Columns to exclude from model inputs"""
        pass

    @abc.abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data pre-model run"""
        pass

    @abc.abstractmethod
    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process data post-model run"""
        pass

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
            model = self.model(input)

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
                    subfolder=self.output_subfolder,
                    filename=self.filename(self.model_id),
                    is_prod=self.is_prod,
                )

        else:
            print(f'Dataframe is empty. Check if data is missing: {self.start_date}')
