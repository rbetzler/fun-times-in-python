-- Table: dw.fact_fred_releases

-- DROP TABLE dw.fact_fred_releases;

CREATE TABLE dw.fact_fred_releases
(
  release_id      numeric,
  link            text,
  name            text,
  press_release   text,
  realtime_end    date,
  realtime_start  date,
  created_at      timestamp without time zone,
  updated_at      timestamp without time zone,
  source_id       numeric
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw.fact_fred_releases
  OWNER TO postgres;
