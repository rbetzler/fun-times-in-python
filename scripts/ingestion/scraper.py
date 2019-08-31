import abc
import bs4
import time
import datetime
import requests
import pandas as pd
import concurrent.futures
from sqlalchemy import create_engine
from scripts.utilities import utils


class WebScraper(abc.ABC):
    def __init__(self,
                 run_date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'),
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d')):
        self.db_connection = utils.DW_STOCKS
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date


    @property
    def get_urls(self) -> pd.DataFrame:
        if self.sql_file is not None:
            urls = self.get_urls_from_db
        else:
            urls = self.one_url
        return urls

    @property
    def one_url(self) -> pd.DataFrame:
        return pd.DataFrame

    @property
    def sql_file(self) -> str:
        return None

    @property
    def query(self) -> str:
        query = open(self.sql_file).read()
        return query

    @property
    def get_urls_from_db(self) -> pd.DataFrame:
        urls = utils.query_db(query=self.query)
        return urls

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
        return soup

    def parse(self) -> pd.DataFrame:
        pass

    def parallelize(self, url) -> pd.DataFrame:
        soup = self.retrieve_web_page(url)
        df = self.parse(soup)
        time.sleep(self.len_of_pause)
        return df

    def execute(self):
        urls = self.get_urls
        df = self.parallel_output
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cores)
        future_to_url = {executor.submit(self.parallelize, row.values[0]): row for idx, row in urls.iterrows()}

        for future in concurrent.futures.as_completed(future_to_url):
            df = pd.concat([df, future.result()], sort=False)

        if self.place_raw_file:
            df.to_csv(self.export_file_path, index=self.place_with_index)

        if self.load_to_db:
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=self.index)
