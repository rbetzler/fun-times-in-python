import datetime
import pandas as pd
import scripts.ingestion.scraper as scraper


class TenKsScraper(scraper.WebScraper):
    @property
    def job_name(self) -> str:
        return 'edgar_filings_ten_qs'

    @property
    def urls_query(self) -> str:
        query = "select distinct company_name, filing_type, date, url from edgar.filings where cik_code = '320193' "\
                "and filing_type = '10-Q' order by date desc limit 1;"
        return query

    def format_urls(self, idx, row) -> tuple:
        url_base = 'https://www.sec.gov/Archives/'
        url = url_base + row['url'].strip()
        company_name = row['company_name'].lower().replace(' ', '_')
        filing_type = row['filing_type'].lower().replace('-', '_')
        url_name = company_name + '_' + filing_type
        return url, url_name

    @property
    def load_to_db(self) -> bool:
        return True

    @property
    def table(self) -> str:
        return 'ten_qs'

    @property
    def schema(self) -> str:
        return 'edgar'

    def parse(self, soup) -> pd.DataFrame:
        soup = soup

        s = str(soup)

        tables = soup.find_all('table')
        table = tables[10].find_all('tr')

        return df


if __name__ == '__main__':
    TenKsScraper().execute()
