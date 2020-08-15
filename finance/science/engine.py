import abc
import datetime
import pandas as pd

from finance.utilities import utils
from finance.science.utilities import modeling_utils


class Engine(abc.ABC):
    def __init__(
            self,
            run_datetime: datetime.datetime = datetime.datetime.utcnow(),
            start_date: datetime.datetime = datetime.date(year=2000, month=1, day=1),
            n_days: int = 1000,
            is_prod: bool = False,
            archive_files: bool = False,
            is_training_run: bool = False,
    ):
        self.run_datetime = run_datetime
        self.start_date = start_date
        self.end_date = start_date + datetime.timedelta(days=int(n_days))
        self._is_prod = is_prod
        self._archive_files = archive_files
        self._is_training_run = is_training_run

    @property
    def is_prod(self) -> bool:
        """Whether the model is production or not"""
        return self._is_prod

    @property
    def archive_files(self) -> bool:
        """Whether to save output files"""
        return self._archive_files

    @property
    def is_training_run(self) -> bool:
        """Whether the model will be trained or not"""
        return self._is_training_run

    @property
    def location(self) -> str:
        """String representation of whether the model will be trained or not"""
        return 'prod' if self.is_prod else 'dev'

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Model id for filename(s)"""
        pass

    @property
    def filename(self) -> str:
        """Filename for archiving results"""
        return f"{self.model_id}_{self.run_datetime.strftime('%Y%m%d%H%M%S')}"

    @property
    def trained_model_filepath(self) -> str:
        """Filepath from where to load and to where to save a trained model"""
        return f'/usr/src/app/audit/science/{self.location}/models/{self.model_id}'

    @property
    @abc.abstractmethod
    def query(self) -> str:
        """Query to retrieve raw data"""
        pass

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

    @abc.abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process data pre-model run"""
        return df

    @abc.abstractmethod
    def run_model(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run model"""
        return df

    @abc.abstractmethod
    def postprocess_data(
            self,
            input: pd.DataFrame,
            output: pd.DataFrame,
    ) -> pd.DataFrame:
        """Process data post-model run"""
        return output

    def execute(self):
        print(f'''
        Timestamp: {datetime.datetime.utcnow()}
        Model ID: {self.model_id}
        Location: {self.location}
        Training: {self.is_training_run}
        Archive: {self.archive_files}
        Start Date: {self.start_date}
        End Date: {self.end_date}
        ''')

        print(f'Getting raw data {datetime.datetime.utcnow()}')
        df = utils.query_db(query=self.query)

        print(f'Processing raw data {datetime.datetime.utcnow()}')
        input = self.preprocess_data(df)

        print(f'Running model {datetime.datetime.utcnow()}')
        output = self.run_model(input)
        if self.archive_files:
            print(f'Saving model predictions to {self.location} {datetime.datetime.utcnow()}')
            modeling_utils.save_file(
                df=output,
                subfolder='predictions',
                filename=self.filename,
                is_prod=self.is_prod,
            )

        trades = self.postprocess_data(input=input, output=output)
        if self.archive_files:
            print(f'Saving trades to {self.location} {datetime.datetime.utcnow()}')
            modeling_utils.save_file(
                df=trades,
                subfolder='trades',
                filename=self.filename,
                is_prod=self.is_prod,
            )
