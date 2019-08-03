import pandas as pd
import scripts.web_scraping.scraper as scraper

class EdgarFileTypesScraper(scraper.WebScraper):

    @property
    def base_url(self) -> str:
        url = 'https://www.sec.gov/forms'
        return pd.DataFrame([url])

    @property
    def drop_raw_file(self) -> bool:
        return True

    @property
    def file_path(self) -> str:
        return '/Users/rickbetzler/Desktop/stock_test.csv'

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return 'dim_edgar_file_types'

    @property
    def schema(self) -> str:
        return DbSchemas().dw_stocks

    def parse(self, soup) -> pd.DataFrame:

        # Use tags to get file types and descriptions
        soup_file_types = soup.find_all('td', {'class' : 'release-number-content views-field views-field-field-release-number is-active'})
        soup_descriptions = soup.find_all('td', {'class' : 'display-title-content views-field views-field-field-display-title'})

        # Load into lists
        file_types = []
        for row in soup_file_types:
            file_types.append(row.get_text().replace('Number:', '').strip())

        file_descriptions = []
        for row in soup_descriptions:
            file_descriptions.append(row.get_text().replace('Description:', '').strip())

        # Convert lists to pandas
        df = pd.DataFrame({
            'file_type' : file_types,
            'description' : file_descriptions,
            'created_at' : pd.Timestamp.now()
            })

        return df


if __name__ == '__main__':
    EdgarFileTypesScraper().execute()

