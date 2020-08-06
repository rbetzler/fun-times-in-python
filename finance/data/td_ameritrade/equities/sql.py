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
            create table if not exists td.equities (
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
                partition by range (market_datetime);

            create table if not exists td.equities_1980 partition of td.equities for values from ('1980-01-01') to ('1990-01-01');
            create table if not exists td.equities_1990 partition of td.equities for values from ('1990-01-01') to ('1995-01-01');
            create table if not exists td.equities_1995 partition of td.equities for values from ('1995-01-01') to ('2000-01-01');
            create table if not exists td.equities_2000 partition of td.equities for values from ('2000-01-01') to ('2001-01-01');
            create table if not exists td.equities_2001 partition of td.equities for values from ('2001-01-01') to ('2002-01-01');
            create table if not exists td.equities_2002 partition of td.equities for values from ('2002-01-01') to ('2003-01-01');
            create table if not exists td.equities_2003 partition of td.equities for values from ('2003-01-01') to ('2004-01-01');
            create table if not exists td.equities_2004 partition of td.equities for values from ('2004-01-01') to ('2005-01-01');
            create table if not exists td.equities_2005 partition of td.equities for values from ('2005-01-01') to ('2006-01-01');
            create table if not exists td.equities_2006 partition of td.equities for values from ('2006-01-01') to ('2007-01-01');
            create table if not exists td.equities_2007 partition of td.equities for values from ('2007-01-01') to ('2008-01-01');
            create table if not exists td.equities_2008 partition of td.equities for values from ('2008-01-01') to ('2009-01-01');
            create table if not exists td.equities_2009 partition of td.equities for values from ('2009-01-01') to ('2010-01-01');
            create table if not exists td.equities_2010 partition of td.equities for values from ('2010-01-01') to ('2011-01-01');
            create table if not exists td.equities_2011 partition of td.equities for values from ('2011-01-01') to ('2012-01-01');
            create table if not exists td.equities_2012 partition of td.equities for values from ('2012-01-01') to ('2013-01-01');
            create table if not exists td.equities_2013 partition of td.equities for values from ('2013-01-01') to ('2014-01-01');
            create table if not exists td.equities_2014 partition of td.equities for values from ('2014-01-01') to ('2015-01-01');
            create table if not exists td.equities_2015 partition of td.equities for values from ('2015-01-01') to ('2016-01-01');
            create table if not exists td.equities_2016 partition of td.equities for values from ('2016-01-01') to ('2017-01-01');
            create table if not exists td.equities_2017 partition of td.equities for values from ('2017-01-01') to ('2018-01-01');
            create table if not exists td.equities_2018 partition of td.equities for values from ('2018-01-01') to ('2019-01-01');
            create table if not exists td.equities_2019 partition of td.equities for values from ('2019-01-01') to ('2020-01-01');
            create table if not exists td.equities_2020 partition of td.equities for values from ('2020-01-01') to ('2021-01-01');
            create table if not exists td.equities_2021 partition of td.equities for values from ('2021-01-01') to ('2022-01-01');
            create table if not exists td.equities_2022 partition of td.equities for values from ('2022-01-01') to ('2023-01-01');
            create table if not exists td.equities_2023 partition of td.equities for values from ('2023-01-01') to ('2024-01-01');
            create table if not exists td.equities_2024 partition of td.equities for values from ('2024-01-01') to ('2025-01-01');

            create table if not exists td.equities_raw (
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
                partition by range (market_datetime);

            create table if not exists td.equities_raw_1980 partition of td.equities_raw for values from ('1980-01-01') to ('1990-01-01');
            create table if not exists td.equities_raw_1990 partition of td.equities_raw for values from ('1990-01-01') to ('1995-01-01');
            create table if not exists td.equities_raw_1995 partition of td.equities_raw for values from ('1995-01-01') to ('2000-01-01');
            create table if not exists td.equities_raw_2000 partition of td.equities_raw for values from ('2000-01-01') to ('2001-01-01');
            create table if not exists td.equities_raw_2001 partition of td.equities_raw for values from ('2001-01-01') to ('2002-01-01');
            create table if not exists td.equities_raw_2002 partition of td.equities_raw for values from ('2002-01-01') to ('2003-01-01');
            create table if not exists td.equities_raw_2003 partition of td.equities_raw for values from ('2003-01-01') to ('2004-01-01');
            create table if not exists td.equities_raw_2004 partition of td.equities_raw for values from ('2004-01-01') to ('2005-01-01');
            create table if not exists td.equities_raw_2005 partition of td.equities_raw for values from ('2005-01-01') to ('2006-01-01');
            create table if not exists td.equities_raw_2006 partition of td.equities_raw for values from ('2006-01-01') to ('2007-01-01');
            create table if not exists td.equities_raw_2007 partition of td.equities_raw for values from ('2007-01-01') to ('2008-01-01');
            create table if not exists td.equities_raw_2008 partition of td.equities_raw for values from ('2008-01-01') to ('2009-01-01');
            create table if not exists td.equities_raw_2009 partition of td.equities_raw for values from ('2009-01-01') to ('2010-01-01');
            create table if not exists td.equities_raw_2010 partition of td.equities_raw for values from ('2010-01-01') to ('2011-01-01');
            create table if not exists td.equities_raw_2011 partition of td.equities_raw for values from ('2011-01-01') to ('2012-01-01');
            create table if not exists td.equities_raw_2012 partition of td.equities_raw for values from ('2012-01-01') to ('2013-01-01');
            create table if not exists td.equities_raw_2013 partition of td.equities_raw for values from ('2013-01-01') to ('2014-01-01');
            create table if not exists td.equities_raw_2014 partition of td.equities_raw for values from ('2014-01-01') to ('2015-01-01');
            create table if not exists td.equities_raw_2015 partition of td.equities_raw for values from ('2015-01-01') to ('2016-01-01');
            create table if not exists td.equities_raw_2016 partition of td.equities_raw for values from ('2016-01-01') to ('2017-01-01');
            create table if not exists td.equities_raw_2017 partition of td.equities_raw for values from ('2017-01-01') to ('2018-01-01');
            create table if not exists td.equities_raw_2018 partition of td.equities_raw for values from ('2018-01-01') to ('2019-01-01');
            create table if not exists td.equities_raw_2019 partition of td.equities_raw for values from ('2019-01-01') to ('2020-01-01');
            create table if not exists td.equities_raw_2020 partition of td.equities_raw for values from ('2020-01-01') to ('2021-01-01');
            create table if not exists td.equities_raw_2021 partition of td.equities_raw for values from ('2021-01-01') to ('2022-01-01');
            create table if not exists td.equities_raw_2022 partition of td.equities_raw for values from ('2022-01-01') to ('2023-01-01');
            create table if not exists td.equities_raw_2023 partition of td.equities_raw for values from ('2023-01-01') to ('2024-01-01');
            create table if not exists td.equities_raw_2024 partition of td.equities_raw for values from ('2024-01-01') to ('2025-01-01');

            create index if not exists equities_symbol_idx on td.equities (symbol);
            create index if not exists equities_market_datetime_idx on td.equities (market_datetime);
            '''
        return ddl

    @property
    def sql_script(self) -> str:
        script = '''
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
                    , market_datetime::date as market_datetime
                    , file_datetime
                    , ingest_datetime
                from partitioned
                where rn = 1
                );
            '''
        return script


if __name__ == '__main__':
    TdEquitiesSQLRunner().execute()
