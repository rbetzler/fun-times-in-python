-- Table: dw.fact_fred_sources

-- DROP TABLE dw.fact_fred_sources;

CREATE TABLE IF NOT EXISTS dw_stocks.fact_fred_sources
(
  source_id         numeric,
  link              text,
  name              text,
  realtime_end      date,
  realtime_start    date,
  dw_created_at     timestamp without time zone,
  dw_updated_at     timestamp without time zone
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw_stocks.fact_fred_sources
  OWNER TO postgres;
