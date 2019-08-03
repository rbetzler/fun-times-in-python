import abc
import bs4
import datetime
import requests
import pandas as pd
import concurrent.futures
from sqlalchemy import create_engine
from scripts.utilities.db_utilities import ConnectionStrings, DbSchemas


class WebScraper(abc.ABC):

    def __init__(self):

        self.db_connection = ConnectionStrings().postgres_dw_stocks

    @property
    def base_url(self) -> pd.DataFrame:
        return pd.DataFrame

    @property
    def urls_in_db(self) -> bool:
        return False

    @property
    def sql_file(self) -> str:
        return ''

    @property
    def query(self) -> str:
        query = open(self.sql_file).read()
        return query

    @property
    def get_urls_from_db(self) -> pd.DataFrame:
        conn = psycopg2.connect(self.db_connection)
        urls = pd.read_sql(self.query, conn)
        return urls

    @property
    def drop_raw_file(self) -> bool:
        return False

    @property
    def file_path(self) -> str:
        return ''

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return ''

    @property
    def schema(self) -> str:
        return DbSchemas().dw_stocks

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
    def n_batches(self) -> int:
        return 1

    def batch_size(self, urls) -> int:
        n = int(len(urls)/self.n_batches)
        return n

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame(columns=['file_type', 'description', 'created_at'])

    @property
    def n_cores(self) -> int:
        return 1

    def retreive_web_page(self, url) -> bs4.BeautifulSoup:
        raw_html = requests.get(url).text
        soup = bs4.BeautifulSoup(raw_html, features="html.parser")
        return soup

    def parse(self) -> pd.DataFrame:
        pass

    def parallelize(self, url) -> pd.DataFrame:
        soup = self.retreive_web_page(url)
        df = self.parse(soup)
        return df

    def execute(self):
        if self.urls_in_db:
            urls = self.get_urls_from_db
        else: urls = self.base_url

        df = self.parallel_output

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_cores)
        future_to_url = {executor.submit(self.parallelize, row.values[0]): row for idx, row in urls.iterrows()}
        for future in concurrent.futures.as_completed(future_to_url):
            df = pd.concat([df, future.result()])

        if self.drop_raw_file:
            df.to_csv(self.file_path)

        if self.load_to_db:
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=self.index)
