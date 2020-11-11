import abc
import datetime
import pandas as pd

from finance.utilities import utils
from finance.science.utilities import modeling_utils


class Decisioner(abc.ABC):
    def __init__(
            self,
            model_id: str,
            run_datetime: datetime.datetime = datetime.datetime.utcnow(),
            start_date: datetime.datetime = datetime.date(year=2000, month=1, day=1),
            n_days: int = 1000,
            is_prod: bool = False,
    ):
        self.model_id = model_id
        self.run_datetime = run_datetime
        self.start_date = start_date
        self.end_date = start_date + datetime.timedelta(days=int(n_days))
        self._is_prod = is_prod

    @property
    @abc.abstractmethod
    def decisioner_id(self) -> str:
        """Unique id for optimization method"""
        pass

    @property
    def is_prod(self) -> bool:
        """Whether the model is production or not"""
        return self._is_prod

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
        End Date: {self.end_date}
        ''')

        print(f'Getting raw data {datetime.datetime.utcnow()}')
        input = utils.query_db(query=self.query)

        print(f'Running optimization {datetime.datetime.utcnow()}')
        output = self.decision(input)
        output['decisioner_id'] = self.decisioner_id

        print(f'Saving output to {self.location} {datetime.datetime.utcnow()}')
        modeling_utils.save_file(
            df=output,
            subfolder='decisions',
            filename=self.filename,
            is_prod=self.is_prod,
        )
