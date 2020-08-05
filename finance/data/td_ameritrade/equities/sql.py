from finance.data import sql


class TdEquitiesSQLRunner(sql.SQLRunner):
    @property
    def table_name(self) -> str:
        return 'equities'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = '''
            CREATE TABLE IF NOT EXISTS td.equities (
                symbol                  text,
                open                    numeric(20,6),
                high                    numeric(20,6),
                low                     numeric(20,6),
                close                   numeric(20,6),
                volume                  numeric(20,6),
                market_datetime_epoch   text,
                empty                   boolean,
                market_datetime         timestamp without time zone,
                file_datetime           timestamp without time zone,
                ingest_datetime         timestamp without time zone
            )
                PARTITION BY RANGE (market_datetime);

            CREATE TABLE IF NOT EXISTS td.equities_1980 PARTITION OF td.equities FOR VALUES FROM ('1980-01-01') TO ('1990-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_1990 PARTITION OF td.equities FOR VALUES FROM ('1990-01-01') TO ('1995-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_1995 PARTITION OF td.equities FOR VALUES FROM ('1995-01-01') TO ('2000-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2000 PARTITION OF td.equities FOR VALUES FROM ('2000-01-01') TO ('2001-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2001 PARTITION OF td.equities FOR VALUES FROM ('2001-01-01') TO ('2002-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2002 PARTITION OF td.equities FOR VALUES FROM ('2002-01-01') TO ('2003-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2003 PARTITION OF td.equities FOR VALUES FROM ('2003-01-01') TO ('2004-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2004 PARTITION OF td.equities FOR VALUES FROM ('2004-01-01') TO ('2005-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2005 PARTITION OF td.equities FOR VALUES FROM ('2005-01-01') TO ('2006-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2006 PARTITION OF td.equities FOR VALUES FROM ('2006-01-01') TO ('2007-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2007 PARTITION OF td.equities FOR VALUES FROM ('2007-01-01') TO ('2008-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2008 PARTITION OF td.equities FOR VALUES FROM ('2008-01-01') TO ('2009-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2009 PARTITION OF td.equities FOR VALUES FROM ('2009-01-01') TO ('2010-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2010 PARTITION OF td.equities FOR VALUES FROM ('2010-01-01') TO ('2011-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2011 PARTITION OF td.equities FOR VALUES FROM ('2011-01-01') TO ('2012-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2012 PARTITION OF td.equities FOR VALUES FROM ('2012-01-01') TO ('2013-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2013 PARTITION OF td.equities FOR VALUES FROM ('2013-01-01') TO ('2014-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2014 PARTITION OF td.equities FOR VALUES FROM ('2014-01-01') TO ('2015-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2015 PARTITION OF td.equities FOR VALUES FROM ('2015-01-01') TO ('2016-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2016 PARTITION OF td.equities FOR VALUES FROM ('2016-01-01') TO ('2017-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2017 PARTITION OF td.equities FOR VALUES FROM ('2017-01-01') TO ('2018-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2018 PARTITION OF td.equities FOR VALUES FROM ('2018-01-01') TO ('2019-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2019 PARTITION OF td.equities FOR VALUES FROM ('2019-01-01') TO ('2020-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2020 PARTITION OF td.equities FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2021 PARTITION OF td.equities FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2022 PARTITION OF td.equities FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2023 PARTITION OF td.equities FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2024 PARTITION OF td.equities FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

            CREATE TABLE IF NOT EXISTS td.equities_raw (LIKE td.equities);
            '''
        return ddl

    @property
    def sql_script(self) -> str:
        script = '''
            drop index if exists equities_symbol_idx;

            truncate td.equities;

            insert into td.equities (
                with partitioned as (
                    select *
                        , row_number() over(partition by symbol, market_datetime order by file_datetime desc) as rn
                    from td.equities_raw
                    )
                select
                    symbol
                    , open
                    , high
                    , low
                    , close
                    , volume
                    , market_datetime_epoch
                    , empty
                    , market_datetime
                    , file_datetime
                    , ingest_datetime
                from partitioned
                where rn = 1
                );

            create index if not exists equities_symbol_idx on td.equities (symbol);
            '''
        return script


if __name__ == '__main__':
    TdEquitiesSQLRunner().execute()
