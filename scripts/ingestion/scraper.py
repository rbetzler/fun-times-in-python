import abc
import bs4
import time
import datetime
import requests
import pandas as pd
import concurrent.futures
from sqlalchemy import create_engine
from scripts.utilities import utils
import zlib


class WebScraper(abc.ABC):
    def __init__(self,
                 run_datetime=datetime.datetime.now()):
        self.db_connection = utils.DW_STOCKS
        self.run_datetime = run_datetime.strftime('%Y-%m-%d-%H-%M')
        self.ingest_datetime = run_datetime.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def job_name(self) -> str:
        return ''

    @property
    def batch_name(self) -> str:
        return 'batch'

    @property
    def get_urls(self) -> pd.DataFrame:
        if self.urls_query:
            urls = self.get_urls_from_db
        else:
            urls = self.py_urls
        return urls

    @property
    def py_urls(self) -> pd.DataFrame:
        return pd.DataFrame

    @property
    def urls_query(self) -> str:
        return ''

    @property
    def get_urls_from_db(self) -> pd.DataFrame:
        urls = []
        names = []
        params = utils.query_db(query=self.urls_query)
        for idx, row in params.iterrows():
            url = self.format_urls(idx, row)
            urls.append(url[0])
            names.append(url[1])
        df = pd.DataFrame(data=urls, index=names)
        return df

    def format_urls(self, idx, row) -> tuple:
        return ()

    @property
    def place_raw_file(self) -> bool:
        return False

    @property
    def place_batch_file(self) -> bool:
        return False

    @property
    def export_folder(self) -> str:
        return ''

    def export_file_path(self, batch) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
                    + batch + '_' \
                    + self.run_date \
                    + self.export_file_type
        return file_path

    @property
    def export_file_name(self) -> str:
        return ''

    @property
    def export_file_type(self) -> str:
        return '.csv'

    @property
    def export_file_path(self, batch) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
                    + batch + '_' \
                    + self.run_datetime \
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
    def insert_audit_record(self):
        query = f" INSERT INTO audit.ingest_load_times" \
                + f" (schema_name, table_name, job_name, ingest_datetime)" \
                + f" VALUES ('{self.schema}', '{self.table}', '{self.job_name}', '{self.ingest_datetime}')"
        utils.insert_record(query=query)
        return

    @property
    def append_to_table(self) -> str:
        return 'append'

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def n_cores(self) -> int:
        return 1

    @property
    def request_type(self) -> str:
        return 'text'

    @property
    def len_of_pause(self) -> int:
        return 0

    def retrieve_web_page(self, url) -> bs4.BeautifulSoup:
        if self.request_type == 'text':
            raw_html = requests.get(url).text
            soup = bs4.BeautifulSoup(raw_html, features="html.parser")
        elif self.request_type == 'json':
            soup = requests.get(url).json()
        elif self.request_type == 'gz':
            soup = zlib.decompress(requests.get(url).content, 16+zlib.MAX_WBITS)
        return soup

    def parse(self) -> pd.DataFrame:
        pass

    def parallelize(self, url) -> pd.DataFrame:
        soup = self.retrieve_web_page(url[0])
        df = self.parse(soup)

        if self.place_raw_file:
            df.to_csv(self.export_file_path(url[1][0]), index=False)

        time.sleep(self.len_of_pause)
        return df

    def execute(self):
        urls = self.get_urls
        df = self.parallel_output
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cores)
        future_to_url = {executor.submit(self.parallelize, row): row for idx, row in urls.iterrows()}

        for future in concurrent.futures.as_completed(future_to_url):
            df = pd.concat([df, future.result()], sort=False)

        if self.place_batch_file:
            df.to_csv(self.export_file_path(self.batch_name), index=False)

        if self.load_to_db:
            if 'dw_created_at' not in df:
                df['dw_created_at'] = self.ingest_datetime
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=False)

            self.insert_audit_record
