import abc
import datetime
import pandas as pd

from finance.utilities import utils
from finance.science.utilities import modeling_utils


class Engine(abc.ABC):
    def __init__(
            self,
            run_datetime=datetime.datetime.utcnow().replace(day=13),
    ):
        self.run_datetime = run_datetime
        self.run_datetime_str = run_datetime.strftime('%Y%m%d%H%M%S')

    @property
    def archive_files(self) -> bool:
        """Whether to save output files"""
        return False

    @property
    def is_prod(self) -> bool:
        """Whether the model is production or not"""
        return False

    @property
    @abc.abstractmethod
    def query(self) -> str:
        """Query to retrieve raw data"""
        pass

    @abc.abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data pre-model run"""
        return df

    @property
    @abc.abstractmethod
    def target_column(self) -> str:
        """Column to target when training and predicting"""
        pass

    @property
    @abc.abstractmethod
    def columns_to_ignore(self) -> list:
        """Columns to ignore when running the model"""
        pass

    @property
    @abc.abstractmethod
    def is_training_run(self) -> bool:
        """Whether or not the model will be trained"""
        pass

    @property
    @abc.abstractmethod
    def trained_model_filepath(self) -> str:
        """Filpath from where to load and to where to save a trained model"""
        pass

    @abc.abstractmethod
    def run_model(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run model"""
        return df

    @abc.abstractmethod
    def post_process_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process data post-model run"""
        return output

    def execute(self):
        location = 'prod' if self.is_prod else 'dev'
        print(f'Running in {location} {datetime.datetime.utcnow()}')

        print(f'Getting raw data {datetime.datetime.utcnow()}')
        df = utils.query_db(query=self.query)

        print(f'Processing raw data {datetime.datetime.utcnow()}')
        input = self.preprocess_data(df)

        print(f'Running model {datetime.datetime.utcnow()}')
        output = self.run_model(input)
        if self.archive_files:
            print(f'Saving model predictions to {location} {datetime.datetime.utcnow()}')
            modeling_utils.save_file(
                df=output,
                subfolder='predictions',
                filename=self.run_datetime_str,
                is_prod=self.is_prod,
            )

        trades = self.post_process_data(input=input, output=output)
        if self.archive_files:
            print(f'Saving trades to {location} {datetime.datetime.utcnow()}')
            modeling_utils.save_file(
                df=trades,
                subfolder='trades',
                filename=self.run_datetime_str,
                is_prod=self.is_prod,
            )
