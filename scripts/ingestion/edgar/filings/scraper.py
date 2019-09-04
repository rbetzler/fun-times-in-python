import datetime
import pandas as pd
import scripts.ingestion.scraper as scraper


class FilingsScraper(scraper.WebScraper):
    def __init__(self, years=None):
        super().__init__()
        self.years = years

    @property
    def job_name(self) -> str:
        return 'filings'

    @property
    def py_urls(self) -> pd.DataFrame:
        urls = []
        names = []
        url_prefix = 'https://www.sec.gov/Archives/edgar/full-index'
        url_suffix = 'company.gz'
        quarters = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
        # current_year = datetime.datetime.utcnow().year + 1
        # for year in range(1993, current_year):
        for year in range(self.years[0], self.years[1]):
            for quarter in quarters:
                url = url_prefix + '/' + str(year) + '/' + quarter + '/' + url_suffix
                urls.append(url)
                name = 'company_index_' + str(year) + '_' + quarter
                names.append(name)
        df = pd.DataFrame(data=urls, index=names)
        return df

    @property
    def load_to_db(self) -> bool:
        return True

    @property
    def table(self) -> str:
        return 'filings'

    @property
    def schema(self) -> str:
        return 'edgar'

    @property
    def request_type(self) -> str:
        return 'gz'

    def parse(self, soup) -> pd.DataFrame:
        soup = str(soup)
        file_header = f'\\n---------------------------------------------------------------------' \
                      '------------------------------------------------------------------------\\n'
        soup = soup[soup.find(file_header)+len(file_header):]
        df = pd.DataFrame(soup.split(r'\n'), columns=['rec'])
        df['splits'] = df['rec'].str.split(r'\s{2,}')

        text = df['rec'].str.rpartition('edgar/data')
        urls = text[1] + text[2]
        text = text[0].str.strip().str.rpartition()
        dates = text[2]
        text = text[0].str.strip().str.rpartition()
        cik_codes = text[2]
        text = text[0].str.strip().str.rpartition()
        filing_types = text[2]
        company_names = text[0].str.strip()

        df = pd.DataFrame({
            'company_name': company_names,
            'filing_type': filing_types,
            'cik_code': cik_codes,
            'date': dates,
            'url': urls
        })

        df = df[df['url'] != '"']
        df = df[df['company_name'] != '']
        return df


if __name__ == '__main__':
    for year in range(2017, 2018):
        print('running year:' + str(year))
        FilingsScraper(years=(year, year+1)).execute()
