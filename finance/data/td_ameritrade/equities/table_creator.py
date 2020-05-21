from finance.data import table_creator


class TdEquitiesTableCreator(table_creator.TableCreator):
    @property
    def table_name(self) -> str:
        return 'equities'

    @property
    def schema_name(self) -> str:
        return 'td'

    @property
    def table_ddl(self) -> str:
        ddl = """
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
            ) PARTITION BY RANGE (market_datetime);
            CREATE TABLE IF NOT EXISTS td.equities_1990 PARTITION OF td.equities 
                FOR VALUES FROM ('1990-01-01') TO ('1995-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_1995 PARTITION OF td.equities 
                FOR VALUES FROM ('1995-01-01') TO ('2000-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2000 PARTITION OF td.equities 
                FOR VALUES FROM ('2000-01-01') TO ('2005-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2005 PARTITION OF td.equities 
                FOR VALUES FROM ('2005-01-01') TO ('2010-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2010 PARTITION OF td.equities 
                FOR VALUES FROM ('2010-01-01') TO ('2015-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_2015 PARTITION OF td.equities 
                FOR VALUES FROM ('2015-01-01') TO ('2020-01-01');    
            CREATE TABLE IF NOT EXISTS td.equities_2020 PARTITION OF td.equities 
                FOR VALUES FROM ('2020-01-01') TO ('2025-01-01');
            CREATE TABLE IF NOT EXISTS td.equities_detail (LIKE td.equities);
            CREATE OR REPLACE VIEW td.equities_view AS (
                with partitioned as (
                    select *, row_number() over(partition by symbol, market_datetime order by file_datetime desc) as rn
                    from td.equities
                )
                select *
                from partitioned
                where rn = 1);
            """
        return ddl

    @property
    def sql_script(self) -> str:
        script = """
        TRUNCATE td.equities;
        INSERT INTO td.equities (
            with partitioned as (
                select *, row_number() over(partition by symbol, market_datetime order by file_datetime desc) as rn
                from td.equities_detail
            )
            select
                symbol,
                open,
                high,
                low,
                close,
                volume,
                market_datetime_epoch,
                empty,
                market_datetime,
                file_datetime,
                ingest_datetime
            from partitioned
            where rn = 1);
        """
        return script


if __name__ == '__main__':
    TdEquitiesTableCreator().execute()
