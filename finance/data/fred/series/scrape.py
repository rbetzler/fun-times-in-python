import pandas as pd
from finance.data import scraper


class FREDSeries(scraper.Caller):
    @property
    def job_name(self) -> str:
        return 'API_FRED'

    @property
    def request_type(self) -> str:
        return 'api'

    @property
    def requests_query(self) -> str:
        return 'select series_id, series_name, category from fred.jobs where is_active;'

    def format_requests(self, row) -> tuple:
        key = row.series_id
        request = f'https://api.stlouisfed.org/fred/series/observations?series_id={key}&api_key={self.api_secret}&file_type=json'
        return key, request

    @property
    def export_folder(self) -> str:
        return 'audit/fred/series/'

    @property
    def export_file_name(self) -> str:
        return 'fred_'

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
            'limit',
        ]
        df = pd.DataFrame()
        for key in keys_to_cols:
            df[key] = [res.get(key)]

        obs = pd.DataFrame(res.get('observations'))
        df = df.merge(obs)
        return df


if __name__ == '__main__':
    FREDSeries().execute()
