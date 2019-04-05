-- Table: dw.fact_yahoo_stocks

-- DROP TABLE dw.fact_yahoo_stocks;

CREATE TABLE dw_stocks.fact_yahoo_stocks
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
  unix_timestamp      text
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw_stocks.fact_yahoo_stocks
  OWNER TO postgres;
