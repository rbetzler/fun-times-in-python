QUERY = """
  CREATE TABLE IF NOT EXISTS yahoo.stocks
  (
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
    dw_created_at       timestamp without time zone,
    dw_updated_at       timestamp without time zone
  );
  """
