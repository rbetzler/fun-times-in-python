import time
import pandas as pd
from finance.utilities import utils
from finance.ingestion import ingestion


class FREDSeriesAPIGrabber(ingestion.Caller):
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
        api_category = row[2]
        return api_call, api_name, api_category

    @property
    def get_calls(self) -> pd.DataFrame:
        calls = []
        names = []
        categories = []
        params = utils.query_db(query=self.calls_query)
        for idx, row in params.iterrows():
            api = self.format_calls(idx, row)
            calls.append(api[0])
            names.append(api[1])
            categories.append(api[2])
        df = pd.DataFrame(data=calls, index=names)
        df[1] = categories
        return df

    # files
    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return 'audit/processed/fred/series/'

    @property
    def export_file_name(self) -> str:
        return 'fred_'

    # parse
    def parse_helper(self, res) -> pd.DataFrame:
        keys_to_cols = ['realtime_start', 'realtime_end', 'observation_start', 'observation_end', 'units',
                        'output_type', 'file_type', 'order_by', 'sort_order', 'count', 'offset', 'limit']
        df = pd.DataFrame()
        for key in keys_to_cols:
            df[key] = [res.get(key)]
        return df

    def parse(self, res, api) -> pd.DataFrame:
        res = res.json()
        df = self.parse_helper(res)
        obs = pd.DataFrame(res.get('observations'))
        df = df.merge(obs)
        # this might simplifiable using json definition
        df['country'] = api[0]
        df['series'] = api[1][1]
        return df

    def parallelize(self, api) -> pd.DataFrame:
        response = self.summon(api[1][0])
        df = self.parse(response, api)

        if bool(self.column_mapping):
            df = df.rename(columns=self.column_mapping)

        if self.place_raw_file:
            export_file_var = api[1][1] + '_' + api[0]
            df.to_csv(self.export_file_path(export_file_var), index=False)

        time.sleep(self.len_of_pause)
        return df


if __name__ == '__main__':
    FREDSeriesAPIGrabber().execute()
