import abc
import datetime
import pandas as pd
from sqlalchemy import create_engine
from finance.utilities import utils


class DerivedTableCreator(abc.ABC):
    def __init__(self,
                 run_datetime=datetime.datetime.now()):
        self.db_connection = utils.DW_STOCKS
        self.ingest_datetime = run_datetime.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def get_data(self) -> pd.DataFrame:
        df = utils.query_db(query=self.query)
        return df

    @property
    def query(self) -> str:
        pass

    @staticmethod
    def apply_logic(df) -> pd.DataFrame:
        return df

    @property
    def load_to_db(self) -> bool:
        return True

    @property
    def table(self) -> str:
        pass

    @property
    def schema(self) -> str:
        pass

    @property
    def db_engine(self):
        return create_engine(self.db_connection)

    def execute(self):
        raw_df = self.get_data
        df = self.apply_logic(raw_df)

        if self.load_to_db:
            if 'ingest_datetime' not in df:
                df['ingest_datetime'] = self.ingest_datetime
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists='append',
                index=False)
