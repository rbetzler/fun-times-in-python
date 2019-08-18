import bs4
import time
import datetime
import requests
import psycopg2
import pandas as pd
from scripts.ingestion import scraper


class Edgar10k(scraper.WebScraper):
    @property
    def one_url(self) -> str:
        url = 'https://www.sec.gov/Archives/edgar/data/320193/000032019318000145/a10-k20189292018.htm'
        return pd.DataFrame([url])

    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return '/Users/rbetzler/Desktop/testing/'

    @property
    def export_file_name(self) -> str:
        return 'edgar_10k_'

    @property
    def export_file_type(self) -> str:
        return '.csv'

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return 'fact_edgar_10k'

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame(columns=['file_type', 'description', 'created_at'])

    @property
    def n_cores(self) -> int:
        return 1

    @property
    def request_type(self) -> str:
        return 'text'

    @property
    def len_of_pause(self) -> int:
        return 4

    def parse(self, soup) -> pd.DataFrame:

        df = soup

        return df


if __name__ == '__main__':
    Edgar10k().execute()
