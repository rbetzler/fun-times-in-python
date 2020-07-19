import abc
import bs4
import time
import datetime
import requests
import pandas as pd
import concurrent.futures
import zlib

from finance.utilities import utils


class Caller(abc.ABC):
    def __init__(
            self,
            run_datetime=datetime.datetime.now(),
            start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
            end_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
            lower_bound=0,
            batch_size=0,
    ):
        self.db_connection = utils.DW_STOCKS
        self.run_datetime = run_datetime.strftime('%Y-%m-%d-%H-%M')
        self.folder_datetime = run_datetime.strftime('%Y%m%d%H%M%S')
        self.ingest_datetime = run_datetime.strftime("%Y-%m-%d %H:%M:%S")

        self.start_date = start_date
        self.end_date = end_date
        self.lower_bound = lower_bound
        self.batch_size = batch_size

    @property
    @abc.abstractmethod
    def job_name(self) -> str:
        pass

    @property
    def api_secret(self) -> str:
        return utils.retrieve_secret(self.job_name)

    @property
    @abc.abstractmethod
    def requests_query(self) -> str:
        pass

    def get_requests(self) -> pd.DataFrame:
        keys = []
        requests_ = []
        params = utils.query_db(query=self.requests_query)
        for row in params.itertuples():
            key, request = self.format_requests(row)
            keys.append(key)
            requests_.append(request)
        df = pd.DataFrame(data=requests_, index=keys, columns=['request'])
        return df

    @abc.abstractmethod
    def format_requests(self, row) -> tuple:
        pass

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

    def export_file_path(self, key) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
                    + key + '_' \
                    + self.folder_datetime \
                    + self.export_file_type
        return file_path

    @property
    def n_workers(self) -> int:
        return 1

    @property
    def request_type(self) -> str:
        return 'text'

    @property
    def len_of_pause(self) -> int:
        return 0

    def get(self, call) -> bs4.BeautifulSoup:
        raw_response = requests.get(call)
        if self.request_type == 'text':
            raw_html = raw_response.text
            response = bs4.BeautifulSoup(raw_html, features="html.parser")
        elif self.request_type == 'json':
            response = raw_response.json()
        elif self.request_type == 'gz':
            response = zlib.decompress(raw_response.content, 16 + zlib.MAX_WBITS)
        elif self.request_type == 'api':
            response = raw_response
        else:
            raise NotImplementedError('API request type not implemented!')
        return response

    @property
    def column_mapping(self) -> dict:
        return {}

    def parse(self, response, key) -> pd.DataFrame:
        pass

    def parallelize(self, idx, row) -> pd.DataFrame:
        response = self.get(row.request)
        df = self.parse(response, idx)

        if bool(self.column_mapping):
            df = df.rename(columns=self.column_mapping)

        df.to_csv(self.export_file_path(idx), index=False)

        time.sleep(self.len_of_pause)
        return df

    def execute(self):
        utils.create_directory(self.export_folder)
        requests_ = self.get_requests()
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers)
        future_to_url = {executor.submit(self.parallelize, idx, row): row for idx, row in requests_.iterrows()}

        failures = []
        for future in concurrent.futures.as_completed(future_to_url):
            if future.exception():
                failures.append(future.exception())

        print(failures)
