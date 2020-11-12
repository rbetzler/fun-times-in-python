from finance.data import sql


class TDStocksSQLRunner(sql.SQLRunner):
    @property
    def table_name(self) -> str:
        return 'stocks'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = '''
            create table td.stocks (
                  market_datetime date
                , symbol varchar
                , open numeric(12,2)
                , high numeric(12,2)
                , low numeric(12,2)
                , close numeric(12,2)
                , volume integer
            )
                partition by range (market_datetime);

            create table if not exists td.stocks_1900 partition of td.stocks for values from ('1900-01-01') to ('1980-01-01');
            create table if not exists td.stocks_1980 partition of td.stocks for values from ('1980-01-01') to ('1990-01-01');
            create table if not exists td.stocks_1990 partition of td.stocks for values from ('1990-01-01') to ('1995-01-01');
            create table if not exists td.stocks_1995 partition of td.stocks for values from ('1995-01-01') to ('2000-01-01');
            create table if not exists td.stocks_2000 partition of td.stocks for values from ('2000-01-01') to ('2001-01-01');
            create table if not exists td.stocks_2001 partition of td.stocks for values from ('2001-01-01') to ('2002-01-01');
            create table if not exists td.stocks_2002 partition of td.stocks for values from ('2002-01-01') to ('2003-01-01');
            create table if not exists td.stocks_2003 partition of td.stocks for values from ('2003-01-01') to ('2004-01-01');
            create table if not exists td.stocks_2004 partition of td.stocks for values from ('2004-01-01') to ('2005-01-01');
            create table if not exists td.stocks_2005 partition of td.stocks for values from ('2005-01-01') to ('2006-01-01');
            create table if not exists td.stocks_2006 partition of td.stocks for values from ('2006-01-01') to ('2007-01-01');
            create table if not exists td.stocks_2007 partition of td.stocks for values from ('2007-01-01') to ('2008-01-01');
            create table if not exists td.stocks_2008 partition of td.stocks for values from ('2008-01-01') to ('2009-01-01');
            create table if not exists td.stocks_2009 partition of td.stocks for values from ('2009-01-01') to ('2010-01-01');
            create table if not exists td.stocks_2010 partition of td.stocks for values from ('2010-01-01') to ('2011-01-01');
            create table if not exists td.stocks_2011 partition of td.stocks for values from ('2011-01-01') to ('2012-01-01');
            create table if not exists td.stocks_2012 partition of td.stocks for values from ('2012-01-01') to ('2013-01-01');
            create table if not exists td.stocks_2013 partition of td.stocks for values from ('2013-01-01') to ('2014-01-01');
            create table if not exists td.stocks_2014 partition of td.stocks for values from ('2014-01-01') to ('2015-01-01');
            create table if not exists td.stocks_2015 partition of td.stocks for values from ('2015-01-01') to ('2016-01-01');
            create table if not exists td.stocks_2016 partition of td.stocks for values from ('2016-01-01') to ('2017-01-01');
            create table if not exists td.stocks_2017 partition of td.stocks for values from ('2017-01-01') to ('2018-01-01');
            create table if not exists td.stocks_2018 partition of td.stocks for values from ('2018-01-01') to ('2019-01-01');
            create table if not exists td.stocks_2019 partition of td.stocks for values from ('2019-01-01') to ('2020-01-01');
            create table if not exists td.stocks_2020 partition of td.stocks for values from ('2020-01-01') to ('2021-01-01');
            create table if not exists td.stocks_2021 partition of td.stocks for values from ('2021-01-01') to ('2022-01-01');
            create table if not exists td.stocks_2022 partition of td.stocks for values from ('2022-01-01') to ('2023-01-01');
            create table if not exists td.stocks_2023 partition of td.stocks for values from ('2023-01-01') to ('2024-01-01');
            create table if not exists td.stocks_2024 partition of td.stocks for values from ('2024-01-01') to ('2025-01-01');

            create index if not exists stocks_symbol_idx on td.stocks (symbol);
            create index if not exists stocks_market_datetime_idx on td.stocks (market_datetime);
            '''
        return ddl

    @property
    def sql_script(self):
        query = '''
            truncate td.stocks;
            insert into td.stocks (
              with
              stocks as (
                select symbol, max(market_datetime) as market_datetime
                from td.stocks
                group by symbol
              )
              , equities as (
                select distinct
                    e.market_datetime
                  , e.symbol
                  , e.open
                  , e.high
                  , e.low
                  , e.close
                  , e.volume
                from td.equities as e
                left join stocks as s
                  on e.symbol = s.symbol
                where (s.symbol is null and s.market_datetime is null)
                    or s.market_datetime < e.market_datetime
              )
              , latest_equities as (
                select symbol, max(market_datetime) as market_datetime
                from equities
                group by symbol
              )
              , quotes as (
                select distinct
                    q.market_datetime
                  , q.symbol
                  , q.open_price as open
                  , q.high_price as high
                  , q.low_price as low
                  , q.regular_market_last_price as close
                  , q.volume
                from td.quotes as q
                left join latest_equities as l
                  on q.symbol = l.symbol
                where (l.symbol is null and l.market_datetime is null)
                    or l.market_datetime < q.market_datetime
              )
              , final as (
                select
                    market_datetime
                  , symbol
                  , open
                  , high
                  , low
                  , close
                  , volume
                from quotes
                union
                select
                    market_datetime
                  , symbol
                  , open
                  , high
                  , low
                  , close
                  , volume
                from equities
              )
              select *
              from final
            );
        '''
        return query


if __name__ == '__main__':
    TDStocksSQLRunner().execute()
