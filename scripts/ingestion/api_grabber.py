import abc
import time
import datetime
import requests
import pandas as pd
import concurrent.futures
from sqlalchemy import create_engine

from scripts.utilities import utils


class APIGrabber(abc.ABC):
    def __init__(self,
                 run_time=datetime.datetime.now(),
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 lower_bound=0,
                 batch_size=0):
        self.db_connection = utils.DW_STOCKS
        self.run_time = run_time
        self.start_date = start_date
        self.end_date = end_date
        self.lower_bound = lower_bound
        self.batch_size = batch_size

    @property
    def run_date(self) -> str:
        return datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')

    @property
    def get_api_calls(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def query(self) -> str:
        return ''

    @property
    def get_call_inputs_from_db(self) -> pd.DataFrame:
        apis = utils.query_db(query=self.query)
        return apis

    @property
    def api_name(self) -> str:
        return ''

    @property
    def api_secret(self) -> str:
        return utils.retrieve_secret(self.api_name)

    @property
    def place_raw_file(self) -> bool:
        return False

    @property
    def place_batch_file(self) -> bool:
        return False

    @property
    def batch_name(self) -> str:
        return 'batch'

    @property
    def place_with_index(self) -> bool:
        return False

    @property
    def export_folder(self) -> str:
        return ''

    @property
    def export_file_name(self) -> str:
        return ''

    @property
    def export_file_type(self) -> str:
        return '.csv'

    def export_file_path(self, batch) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
                    + batch + '_' \
                    + self.run_date \
                    + self.export_file_type
        return file_path

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return ''

    @property
    def schema(self) -> str:
        return ''

    @property
    def db_engine(self) -> str:
        return create_engine(self.db_connection)

    @property
    def append_to_table(self) -> str:
        return 'append'

    @property
    def index(self) -> bool:
        return False

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def column_mapping(self) -> dict:
        return {}

    @property
    def n_workers(self) -> int:
        return 1

    @property
    def len_of_pause(self) -> int:
        return 0

    def call_api(self, call) -> requests.Response:
        api_response = requests.get(call)
        return api_response

    def parse(self) -> pd.DataFrame:
        pass

    def parallelize(self, api) -> pd.DataFrame:
        api_response = self.call_api(api[1][0])
        df = self.parse(api_response)

        if bool(self.column_mapping):
            df = df.rename(columns=self.column_mapping)

        if self.place_raw_file:
            df.to_csv(self.export_file_path(api[0]), index=self.place_with_index)

        time.sleep(self.len_of_pause)
        return df

    def execute(self):
        utils.create_directory(self.export_folder)
        api_calls = self.get_api_calls
        df = self.parallel_output
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers)
        future_to_url = {executor.submit(self.parallelize, row): row for row in api_calls.iterrows()}

        for future in concurrent.futures.as_completed(future_to_url):
            df = pd.concat([df, future.result()], sort=False)

        if self.place_batch_file:
            df.to_csv(self.export_file_path(self.batch_name), index=self.place_with_index)

        if self.load_to_db:
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=self.index)
