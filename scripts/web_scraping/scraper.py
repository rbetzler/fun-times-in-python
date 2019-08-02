import abc
import pandas as pd
import datetime

import bs4
import sys
import requests
import pandas as pd
from sqlalchemy import create_engine
from scripts.utilities.db_utilities import ConnectionStrings, DbSchemas


class WebScraper(abc.ABC):

    def __init__(self):

        self.db_connection = ConnectionStrings().postgres_dw_stocks

    @property
    def base_url(self) -> str:
        return ''

    def url_iterator(self, base_url) -> list:
        return [base_url]

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
        return ''

    @property
    def db_engine(self) -> str:
        return create_engine(self.db_connection)

    @property
    def append_to_table(self) -> bool:
        return 'append'

    @property
    def index(self) -> bool:
        return False

    def retreive_web_page(self, url) -> bs4.BeautifulSoup:
        raw_html = requests.get(url).text
        soup = bs4.BeautifulSoup(raw_html)
        return soup

    def parse(self, soup) -> pd.DataFrame:
        return df

    def execute(self):

        urls = self.url_iterator(self.base_url)

        for url in urls:
            soup = self.retreive_web_page(url)

            df = self.parse(soup)

            if self.drop_raw_file:
                df.to_csv(self.file_path)

            if self.load_to_db:
                df.to_sql(
                    self.table,
                    self.db_engine,
                    schema=self.schema,
                    if_exists=self.append_to_table,
                    index=self.index)
