import pandas as pd
import scripts.ingestion.scraper as scraper


class EdgarFileTypesScraper(scraper.WebScraper):
    @property
    def py_urls(self) -> str:
        url = 'https://www.sec.gov/forms'
        return pd.DataFrame([url])

    @property
    def load_to_db(self) -> bool:
        return True

    @property
    def table(self) -> str:
        return 'file_types'

    @property
    def schema(self) -> str:
        return 'edgar'

    def parse(self, soup) -> pd.DataFrame:

        # Use tags to get file types and descriptions
        soup_file_types = soup.find_all(
            'td', {'class': 'release-number-content views-field views-field-field-release-number is-active'})
        soup_descriptions = soup.find_all(
            'td', {'class': 'display-title-content views-field views-field-field-display-title'})

        # Load into lists
        file_types = []
        for row in soup_file_types:
            file_types.append(row.get_text().replace('Number:', '').strip())

        file_descriptions = []
        for row in soup_descriptions:
            file_descriptions.append(row.get_text().replace('Description:', '').strip())

        # Convert lists to pandas
        df = pd.DataFrame({
            'file_type': file_types,
            'description': file_descriptions
            })

        return df


if __name__ == '__main__':
    EdgarFileTypesScraper().execute()

