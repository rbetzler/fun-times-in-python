import abc
import time
import datetime
import requests
import psycopg2
import pandas as pd
import concurrent.futures
from sqlalchemy import create_engine
from scripts.utilities import db_utilities


class ApiGrabber(abc.ABC):
    def __init__(self,
                 run_date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'),
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d')):
        self.db_connection = db_utilities.DW_STOCKS
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date


    @property
    def get_api_calls(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def query(self) -> str:
        return ''

    @property
    def get_call_inputs_from_db(self) -> pd.DataFrame:
        conn = psycopg2.connect(self.db_connection)
        apis = pd.read_sql(self.query, conn)
        return apis

    @property
    def place_raw_file(self) -> bool:
        return False

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

    @property
    def export_file_path(self) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
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
        return 'td_ameritrade'

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
    def n_cores(self) -> int:
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
        api_response = self.call_api(api)
        df = self.parse(api_response)
        return df

    def execute(self):
        api_calls = self.get_api_calls
        df = self.parallel_output
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cores)
        future_to_url = {executor.submit(self.parallelize, row[0]): row for idx, row in api_calls.iterrows()}

        for future in concurrent.futures.as_completed(future_to_url):
            df = pd.concat([df, future.result()], sort=False)
            # df = df.append(future.result())
            time.sleep(self.len_of_pause)

        if self.place_raw_file:
            df.to_csv(self.export_file_path, index=self.place_with_index)

        if self.load_to_db:
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=self.index)
