import pandas as pd
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
        limit 10000
        """
        return query

    @staticmethod
    def apply_logic(df) -> pd.DataFrame:
        df['position'] = None
        df['profit_loss'] = None
        for symbol in df['symbol'].unique():
            position = 1000
            for idx, row in df[df['symbol'] == symbol].iterrows():
                ma_five = df.loc[idx-5:idx, 'open'].mean()
                ma_twenty = df.loc[idx-20:idx, 'open'].mean()
                if position != 0:
                    if ma_five < ma_twenty:
                        position = 0
                else:
                    if ma_five > ma_twenty:
                        position = row['open']
                df.loc[idx, 'position'] = position
        daily = df[['market_datetime', 'position']].groupby('market_datetime').sum()
        return daily

    @property
    def table(self) -> str:
        return 'benchmarks'

    @property
    def schema(self) -> str:
        return 'evaluation'


if __name__ == '__main__':
    MovingAverageTable().execute()