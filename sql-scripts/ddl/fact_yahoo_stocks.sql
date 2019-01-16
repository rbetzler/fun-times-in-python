-- Table: public.fact_yahoo_stocks

-- DROP TABLE public.fact_yahoo_stocks;

CREATE TABLE public.fact_yahoo_stocks
(
  ticker text,
  date_time timestamp without time zone,
  open numeric,
  high numeric,
  low numeric,
  close numeric,
  adj_close numeric,
  volume bigint,
  dividend numeric,
  split_numerator integer,
  split_denominator integer,
  index text,
  unix_timestamp text
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.fact_yahoo_stocks
  OWNER TO postgres;
COMMENT ON TABLE public.fact_yahoo_stocks
  IS 'Historical stock performances scraped from yahoo finance';
