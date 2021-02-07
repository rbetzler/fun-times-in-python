create table yahoo.sp_index (
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
);

create index if not exists yahoo_sp_file_datetime_idx on td.black_scholes (market_datetime);
