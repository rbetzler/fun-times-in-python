import pandas as pd
from finance.data_science import derived_table_creator


class MovingAverageTable(derived_table_creator.DerivedTableCreator):
    @property
    def query(self) -> str:
        query = """
        with adding_moving_averages as (
        select
            date(market_datetime) as market_datetime,
            symbol,
            open,
            avg(open) over (partition by symbol order by market_datetime 
                rows between 20 preceding and current row) as twenty_day_ma,
            open - avg(open) over (partition by symbol order by market_datetime 
                rows between 20 preceding and current row) as twenty_day_ma_diff,
            open > avg(open) over (partition by symbol order by market_datetime 
                rows between 20 preceding and current row) as twenty_day_ma_flag
        from td.equities_view
        limit 10000),
        adding_row_numbers as (
        select *,
            row_number() over (partition by symbol order by market_datetime) as rn,
            row_number() over (partition by symbol, twenty_day_ma_flag order by market_datetime) as rn_sub,
            row_number() over (partition by symbol, twenty_day_ma_flag order by market_datetime) 
                - row_number() over (partition by symbol order by market_datetime) as rn_diff
        from adding_moving_averages),
        adding_partitions as (
        select *,
            first_value(market_datetime) over (
                partition by symbol, twenty_day_ma_flag, rn_diff 
                order by market_datetime 
                rows between unbounded preceding and unbounded following) as buy_date,
            last_value(market_datetime) over (
                partition by symbol, twenty_day_ma_flag, rn_diff 
                order by market_datetime 
                rows between unbounded preceding and unbounded following) as sell_date,
            first_value(open) over (
                partition by symbol, twenty_day_ma_flag, rn_diff 
                order by market_datetime 
                rows between unbounded preceding and unbounded following) as buy_open,
            last_value(open) over (
                partition by symbol, twenty_day_ma_flag, rn_diff 
                order by market_datetime 
                rows between unbounded preceding and unbounded following) as sell_open
        from adding_row_numbers)
        select distinct
            symbol,
            buy_date,
            sell_date,
            buy_open,
            sell_open,
            (sell_open - buy_open)/ sell_open as profit_loss
        from adding_partitions 
        where open <> 0
        order by symbol, buy_date
        """
        return query

    @staticmethod
    def apply_logic(df) -> pd.DataFrame:
        df['position'] = None
        for symbol in df['symbol'].unique():
            position = 1000
            for idx, row in df[df['symbol'] == symbol].iterrows():
                position = position * (1 + row['profit_loss'])
                df.loc[idx, 'position'] = position
        daily = df[['buy_date', 'position']].groupby('buy_date').sum()
        return daily

    @property
    def table(self) -> str:
        return 'benchmarks'

    @property
    def schema(self) -> str:
        return 'evaluation'


if __name__ == '__main__':
    MovingAverageTable().execute()