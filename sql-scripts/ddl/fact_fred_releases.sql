-- Table: dw.fact_fred_releases

-- DROP TABLE dw.fact_fred_releases;

CREATE TABLE IF NOT EXISTS dw_stocks.fact_fred_releases
(
  release_id      numeric,
  link            text,
  name            text,
  press_release   text,
  realtime_end    date,
  realtime_start  date,
  source_id       numeric,
  dw_created_at   timestamp without time zone,
  dw_updated_at   timestamp without time zone
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw_stocks.fact_fred_releases
  OWNER TO postgres;
