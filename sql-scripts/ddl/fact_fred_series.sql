-- Table: public.fact_fred_series

-- DROP TABLE public.fact_fred_series;

CREATE TABLE public.fact_fred_series
(
  release_id numeric,
  series_id text,
  frequency_long text,
  frequency text,
  group_popularity numeric,
  last_updated timestamp without time zone,
  notes text,
  observation_end_date date,
  observation_start_date date,
  popularity numeric,
  realtime_end date,
  realtime_start date,
  seasonal_adj_long text,
  seasonal_adj text,
  title text,
  units_long text,
  units text,
  created_at timestamp without time zone,
  updated_at timestamp without time zone
)
WITH (
  OIDS=FALSE
);
ALTER TABLE public.fact_fred_series
  OWNER TO postgres;
