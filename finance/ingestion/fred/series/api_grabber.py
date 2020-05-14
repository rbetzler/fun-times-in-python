import pandas as pd
from finance.ingestion import scraper


class FREDSeriesAPIGrabber(scraper.Caller):
    # general
    @property
    def api_name(self) -> str:
        return 'API_FRED'

    @property
    def request_type(self) -> str:
        return 'api'

    # calls
    @property
    def calls_query(self) -> str:
        return "select series_id, series_name, category from fred.jobs where is_active;"

    def format_calls(self, idx, row) -> tuple:
        api_call = 'https://api.stlouisfed.org/fred/series/observations?' \
                   + 'series_id=' + row[0] \
                   + '&api_key=' + self.api_secret \
                   + '&file_type=json'
        api_name = row[1]
        return api_call, api_name

    # files
    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return 'audit/fred/series/'

    @property
    def export_file_name(self) -> str:
        return 'fred_'

    # parse
    def parse(self, res, call) -> pd.DataFrame:
        res = res.json()

        keys_to_cols = [
            'realtime_start',
            'realtime_end',
            'observation_start',
            'observation_end',
            'units',
            'output_type',
            'file_type',
            'order_by',
            'sort_order',
            'count',
            'offset',
            'limit'
        ]
        df = pd.DataFrame()
        for key in keys_to_cols:
            df[key] = [res.get(key)]

        obs = pd.DataFrame(res.get('observations'))
        df = df.merge(obs)
        return df


if __name__ == '__main__':
    FREDSeriesAPIGrabber().execute()
