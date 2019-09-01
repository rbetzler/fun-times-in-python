import pandas as pd
from scripts.ingestion import api_grabber


class FREDInflationAPIGrabber(api_grabber.APIGrabber):
    @property
    def api_calls_query(self) -> str:
        return "select series_id, series_name from fred.jobs where is_active and category = 'inflation';"

    def format_api_calls(self, idx, row) -> tuple:
        api_call = 'https://api.stlouisfed.org/fred/series/observations?' \
                   + 'series_id=' + row[0] \
                   + '&api_key=' + self.api_secret \
                   + '&file_type=json'
        api_name = row[1]
        return api_call, api_name

    @property
    def api_name(self) -> str:
        return 'API_FRED'

    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return 'audit/processed/fred/inflation/'

    @property
    def export_file_name(self) -> str:
        return 'fred_'

    def parse_helper(self, res) -> pd.DataFrame:
        keys_to_cols = ['realtime_start', 'realtime_end', 'observation_start', 'observation_end', 'units',
                        'output_type', 'file_type', 'order_by', 'sort_order', 'count', 'offset', 'limit']
        df = pd.DataFrame()
        for key in keys_to_cols:
            df[key] = [res.get(key)]
        return df

    def parse(self, res) -> pd.DataFrame:
        res = res.json()
        df = self.parse_helper(res)
        obs = pd.DataFrame(res.get('observations'))
        df = df.merge(obs)
        return df


if __name__ == '__main__':
    FREDInflationAPIGrabber().execute()
