STOCKS = '''
    CREATE TABLE yahoo.stocks (
        ticker              text,
        date_time           timestamp without time zone,
        open                numeric,
        high                numeric,
        low                 numeric,
        close               numeric,
        adj_close           numeric,
        volume              bigint,
        dividend            numeric,
        split_numerator     integer,
        split_denominator   integer,
        index               text,
        unix_timestamp      text,
        file_datetime       timestamp without time zone,
        ingest_datetime     timestamp without time zone
    );
    '''

INCOME_STATEMENTS = '''
    CREATE TABLE yahoo.income_statements (
        date                timestamp without time zone,
        ticker              text,
        metric              text,
        val                 numeric(20,6),
        file_datetime       timestamp without time zone,
        ingest_datetime     timestamp without time zone
    );
    '''

BALANCE_SHEETS = '''
    CREATE TABLE yahoo.balance_sheets (
        date                timestamp without time zone,
        ticker              text,
        metric              text,
        val                 numeric(20,6),
        file_datetime       timestamp without time zone,
        ingest_datetime     timestamp without time zone
    );
    '''

CASH_FLOWS = '''
    CREATE TABLE yahoo.cash_flows (
        date                timestamp without time zone,
        ticker              text,
        metric              text,
        val                 numeric(20,6),
        file_datetime       timestamp without time zone,
        ingest_datetime     timestamp without time zone
    );
    '''

SP_INDEX = '''
    CREATE TABLE yahoo.sp_index (
        index_name          text,
        market_datetime     timestamp without time zone,
        open                numeric,
        high                numeric,
        low                 numeric,
        close               numeric,
        adj_close           numeric,
        volume              bigint,
        file_datetime       timestamp without time zone,
        ingest_datetime     timestamp without time zone
    )
    '''
