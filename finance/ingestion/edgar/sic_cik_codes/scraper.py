import pandas as pd
from string import ascii_lowercase
from finance.ingestion import scraper


class SICCIKCodesScraper(scraper.Caller):
    # general
    @property
    def job_name(self) -> str:
        return 'edgar_sic_cik_codes'

    # calls
    @property
    def py_urls(self) -> str:
        url_prefix = 'https://www.sec.gov/divisions/corpfin/organization/cfia-'
        url_suffix = '.htm'
        urls = []
        for letter in ascii_lowercase:
            if letter == 'u':
                break
            url = url_prefix + letter + url_suffix
            urls.append(url)
        for grouped_chars in ['uv', 'wxyz', '123']:
            urls.append(url_prefix + grouped_chars + url_suffix)
        return pd.DataFrame(urls)

    # db
    @property
    def load_to_db(self) -> bool:
        return True

    @property
    def table(self) -> str:
        return 'sic_cik_codes'

    @property
    def schema(self) -> str:
        return 'edgar'

    # parse
    def parse(self, soup) -> pd.DataFrame:
        company_names = []
        ciks = []
        sics = []

        tbls = soup.find_all('table')
        tbl = tbls[2].find('table')
        recs = tbl.find_all('tr')
        for rec in recs:
            rows = rec.find_all('td')
            if rows:
                company_names.append(rows[0].string)
                ciks.append(rows[1].string)
                sics.append(rows[2].string)
        df = pd.DataFrame({
            'company_name': company_names,
            'cik_code': ciks,
            'sic_code': sics
        })
        return df


if __name__ == '__main__':
    SICCIKCodesScraper().execute()

