import abc
import time
import datetime
import requests
import pandas as pd
import concurrent.futures

from typing import List, Tuple
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
    def calls_query(self) -> str:
        """Query to get api call info from dw"""
        pass

    @abc.abstractmethod
    def format_calls(self, row) -> tuple:
        """Format a pandas record into an api call"""
        pass

    def get_calls(self) -> List[Tuple]:
        """Generate a list of api calls (as tuples)"""
        keys_requests = []
        params = utils.query_db(query=self.calls_query)
        for row in params.itertuples():
            key_request = self.format_calls(row)
            keys_requests.append(key_request)
        return keys_requests

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
        return f'{self.export_folder}{self.export_file_name}{key}_{self.folder_datetime}{self.export_file_type}'

    @property
    def n_workers(self) -> int:
        return 1

    @property
    def len_of_pause(self) -> int:
        return 0

    @property
    def column_mapping(self) -> dict:
        return {}

    def parse(
            self,
            response: requests.Response,
            key: str,
    ) -> pd.DataFrame:
        """Parsing logic on an api request"""
        pass

    def get(
            self,
            key: str,
            call: str,
    ):
        """Get data from an api, parse it, write it as a csv"""
        response = requests.get(call)
        df = self.parse(response, key)
        if bool(self.column_mapping):
            df = df.rename(columns=self.column_mapping)
        df.to_csv(self.export_file_path(key), index=False)
        time.sleep(self.len_of_pause)

    def parallel_execute(
            self,
            key: str,
            call: str,
    ):
        """Run the call function; if errs, sleep and rerun"""
        try:
            self.get(key, call)
        except Exception as e:
            print(f'''
                {key} failed with error: {e}
                Sleeping then retrying {key}
                ''')
            time.sleep(self.len_of_pause)
            self.get(key, call)

    def execute(self):
        utils.create_directory(self.export_folder)
        calls = self.get_calls()
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers)
        future_to_url = {
            executor.submit(
                self.parallel_execute,
                key,
                call,
            ): key for key, call in calls
        }
        failures = []
        for future in concurrent.futures.as_completed(future_to_url):
            if future.exception():
                failures.append(future.exception())
        print(failures)
