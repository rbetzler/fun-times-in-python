import pandas as pd
from scripts.ingestion import api_grabber


class FREDSeriesAPIGrabber(api_grabber.APIGrabber):
    @property
    def api_calls_query(self) -> str:
        return "select search from fred.series_searches where is_active ;"

    def format_api_calls(self, idx, row) -> tuple:
        api_call = 'https://api.stlouisfed.org/fred/series/search' \
                   + '?search_text=' + row[0] \
                   + '&api_key=' + self.api_secret \
                   + '&file_type=json'
        api_name = row[0]
        return api_call, api_name

    @property
    def api_name(self) -> str:
        return 'API_FRED'

    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return 'audit/processed/fred/series_searches/'

    @property
    def export_file_name(self) -> str:
        return 'fred_series_searches_'

    def parse(self, res) -> pd.DataFrame:
        res = res.json()
        series = res.get('seriess')
        df = pd.DataFrame(series)
        df = df.sort_values(by='title')
        return df


if __name__ == '__main__':
    FREDSeriesAPIGrabber().execute()
