import datetime
import pandas as pd
import concurrent.futures
from finance.data_science import derived_table_creator


class MovingAverageTable(derived_table_creator.DerivedTableCreator):
    @property
    def query(self) -> str:
        query = """
        select
            date(market_datetime) as market_datetime,
            symbol,
            open
        from td.equities_view
        order by symbol, market_datetime
        """
        return query

    @property
    def n_workers(self) -> int:
        return 10

    @staticmethod
    def calculate_daily_position(df) -> pd.DataFrame:
        position = 1000
        for idx, row in df.iterrows():
            if df.loc[idx - 5:idx, 'open'].mean() > df.loc[idx - 20:idx, 'open'].mean():
                position = row['open']
            df.loc[idx, 'account_value'] = position
        return df

    def apply_logic(self, df) -> pd.DataFrame:
        print('Running parallel job ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        df['account_value'] = None
        temp = []

        dfs = dict(tuple(df.groupby('symbol')))

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers)
        future_to_url = {executor.submit(self.calculate_daily_position, dfs.get(symbol)):
                             dfs.get(symbol) for symbol in dfs}
        for future in concurrent.futures.as_completed(future_to_url):
            temp.append(future.result())

        output = pd.concat(temp, sort=False)

        daily = output[['market_datetime', 'account_value']].groupby('market_datetime').sum()
        daily['model_type'] = 'moving_average_five_twenty_days'
        daily = daily.reset_index()
        print('Finished parallel job ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return daily

    @property
    def table(self) -> str:
        return 'benchmarks'

    @property
    def schema(self) -> str:
        return 'validation'


if __name__ == '__main__':
    MovingAverageTable().execute()