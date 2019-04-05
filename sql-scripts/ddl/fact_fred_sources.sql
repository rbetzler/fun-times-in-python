-- Table: dw.fact_fred_sources

-- DROP TABLE dw.fact_fred_sources;

CREATE TABLE dw_stocks.fact_fred_sources
(
  source_id       numeric,
  link            text,
  name            text,
  realtime_end    date,
  realtime_start  date,
  created_at      timestamp without time zone,
  updated_at      timestamp without time zone
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw_stocks.fact_fred_sources
  OWNER TO postgres;
