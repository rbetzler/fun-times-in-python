import abc
import datetime
import pandas as pd

from science import core
from trading import utils as trading_utils
from utilities import utils

from science.utilities import modeling_utils


class Decisioner(core.Science, abc.ABC):

    @property
    @abc.abstractmethod
    def decisioner_id(self) -> str:
        """Unique id for optimization method"""
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
                filename=self.filename(self.decisioner_id),
                is_prod=self.is_prod,
            )
