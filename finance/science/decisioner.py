import abc
import datetime
import pandas as pd

from finance.trading import utils as trading_utils
from finance.utilities import utils
from finance.science.utilities import modeling_utils


class Decisioner(abc.ABC):
    def __init__(
            self,
            run_datetime: datetime.datetime = datetime.datetime.utcnow(),
            start_date: datetime.datetime = datetime.date(year=2000, month=1, day=1),
            is_prod: bool = False,
            archive_files: bool = False,
    ):
        self.run_datetime = run_datetime
        self.start_date = start_date
        self._is_prod = is_prod
        self._archive_files = archive_files

    @property
    def is_prod(self) -> bool:
        """Whether the model is production or not"""
        return self._is_prod

    @property
    def archive_files(self) -> bool:
        """Whether to save output files"""
        return self._archive_files

    @property
    @abc.abstractmethod
    def model_id(self) -> str:
        """Unique id for prediction method"""
        pass

    @property
    @abc.abstractmethod
    def decisioner_id(self) -> str:
        """Unique id for optimization method"""
        pass

    @property
    def location(self) -> str:
        """String for whether the results will be save in dev or prod"""
        return 'prod' if self.is_prod else 'dev'

    @property
    def filename(self) -> str:
        """Filename for archiving results"""
        return f"{self.decisioner_id}_{self.run_datetime.strftime('%Y%m%d%H%M%S')}"

    @property
    @abc.abstractmethod
    def query(self) -> str:
        """Query to retrieve raw data"""
        pass

    @property
    def available_funds(self) -> int:
        """Get available funds from all TD Accounts"""
        funds = 0
        for account in trading_utils.TDAccounts().get_accounts():
            funds += account.available_funds
        return funds

    @property
    def n_trades(self) -> int:
        """Max number of trades"""
        return 3

    @property
    def max_position_size(self) -> int:
        """Max position size per trade"""
        return 200

    @abc.abstractmethod
    def decision(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decision & optimization step"""
        return df

    def execute(self):
        print(f'''
        Timestamp: {datetime.datetime.utcnow()}
        Optimizer ID: {self.decisioner_id}
        Location: {self.location}
        Start Date: {self.start_date}
        ''')

        print(f'Getting raw data {datetime.datetime.utcnow()}')
        df = utils.query_db(query=self.query)

        print(f'Running optimization {datetime.datetime.utcnow()}')
        output = self.decision(df)
        output['decisioner_id'] = self.decisioner_id

        if self.archive_files:
            print(f'Saving output to {self.location} {datetime.datetime.utcnow()}')
            modeling_utils.save_file(
                df=output,
                subfolder='decisions',
                filename=self.filename,
                is_prod=self.is_prod,
            )
