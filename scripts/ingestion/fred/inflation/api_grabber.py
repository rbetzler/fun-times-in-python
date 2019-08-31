import pandas as pd
from scripts.ingestion import api_grabber


class FREDInflationAPIGrabber(api_grabber.APIGrabber):
    @property
    def query(self) -> str:
        return "select series_id, series_name from fred.jobs where is_active and category = 'inflation'; "

    @property
    def get_api_calls(self) -> pd.DataFrame:
        apis = []
        series = []
        for idx, row in self.get_call_inputs_from_db.iterrows():
            apis.append('https://api.stlouisfed.org/fred/series/observations?'
                        + 'series_id=' + row[0]
                        + '&api_key=' + self.api_secret
                        + '&file_type=json')
            series.append(row[1])
        df = pd.DataFrame(data=apis, index=series)
        return df

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
