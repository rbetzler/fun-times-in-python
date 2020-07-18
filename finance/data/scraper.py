import abc
import bs4
import time
import datetime
import requests
import pandas as pd
import concurrent.futures
from sqlalchemy import create_engine
from sqlalchemy import engine

from finance.utilities import utils
import zlib


class Caller(abc.ABC):
    def __init__(self,
                 run_datetime=datetime.datetime.now(),
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 lower_bound=0,
                 batch_size=0):
        self.db_connection = utils.DW_STOCKS
        self.run_datetime = run_datetime.strftime('%Y-%m-%d-%H-%M')
        self.folder_datetime = run_datetime.strftime('%Y%m%d%H%M%S')
        self.ingest_datetime = run_datetime.strftime("%Y-%m-%d %H:%M:%S")

        self.start_date = start_date
        self.end_date = end_date
        self.lower_bound = lower_bound
        self.batch_size = batch_size

    @property
    def job_name(self) -> str:
        return 'api_caller'

    @property
    def batch_name(self) -> str:
        return 'batch'

    @property
    @abc.abstractmethod
    def api_name(self) -> str:
        pass

    @property
    def api_secret(self) -> str:
        return utils.retrieve_secret(self.api_name)

    @property
    @abc.abstractmethod
    def calls_query(self) -> str:
        pass

    def get_calls_from_db(self) -> pd.DataFrame:
        calls = []
        names = []
        params = utils.query_db(query=self.calls_query)
        for idx, row in params.iterrows():
            call = self.format_calls(idx, row)
            calls.append(call[0])
            names.append(call[1])
        df = pd.DataFrame(data=calls, index=names)
        return df

    @abc.abstractmethod
    def format_calls(self, idx, row) -> tuple:
        pass

    @property
    def place_raw_file(self) -> bool:
        return False

    @property
    def place_batch_file(self) -> bool:
        return False

    @property
    @abc.abstractmethod
    def export_folder(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def export_file_name(self) -> str:
        pass

    @property
    def export_file_type(self) -> str:
        return '.csv'

    def export_file_path(self, batch) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
                    + batch + '_' \
                    + self.folder_datetime \
                    + self.export_file_type
        return file_path

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        pass

    @property
    def schema(self) -> str:
        return ''

    @property
    def db_engine(self) -> engine.Engine:
        return create_engine(self.db_connection)

    def insert_audit_record(self):
        query = f'''
            INSERT INTO audit.ingest_load_times' (schema_name, table_name, job_name, ingest_datetime)'
            VALUES ('{self.schema}', '{self.table}', '{self.job_name}', '{self.ingest_datetime}')
            '''
        utils.insert_record(query=query)

    @property
    def append_to_table(self) -> str:
        return 'append'

    @property
    def n_workers(self) -> int:
        return 1

    @property
    def request_type(self) -> str:
        return 'text'

    @property
    def len_of_pause(self) -> int:
        return 0

    def summon(self, call) -> bs4.BeautifulSoup:
        if self.request_type == 'text':
            raw_html = requests.get(call).text
            response = bs4.BeautifulSoup(raw_html, features="html.parser")
        elif self.request_type == 'json':
            response = requests.get(call).json()
        elif self.request_type == 'gz':
            response = zlib.decompress(requests.get(call).content, 16+zlib.MAX_WBITS)
        elif self.request_type == 'api':
            response = requests.get(call)
        else:
            raise NotImplementedError('API request type not implemented!')
        return response

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def column_mapping(self) -> dict:
        return {}

    def parse(self, response, call) -> pd.DataFrame:
        pass

    def parallelize(self, call) -> pd.DataFrame:
        response = self.summon(call[1].values[0])
        df = self.parse(response, call[0])

        if bool(self.column_mapping):
            df = df.rename(columns=self.column_mapping)

        if self.place_raw_file:
            df.to_csv(self.export_file_path(call[0]), index=False)

        time.sleep(self.len_of_pause)
        return df

    def execute(self):
        utils.create_directory(self.export_folder)
        calls = self.get_calls_from_db()
        df = self.parallel_output
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers)
        future_to_url = {executor.submit(self.parallelize, call): call for call in calls.iterrows()}

        for future in concurrent.futures.as_completed(future_to_url):
            df = pd.concat([df, future.result()], sort=False)

        if self.place_batch_file:
            df.to_csv(self.export_file_path(self.batch_name), index=False)

        if self.load_to_db:
            if 'ingest_datetime' not in df:
                df['ingest_datetime'] = self.ingest_datetime
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=False,
            )

            self.insert_audit_record()
