-- Table: dw.fact_fred_releases

-- DROP TABLE dw.fact_fred_releases;

CREATE TABLE IF NOT EXISTS dw_stocks.dim_edgar_file_types
(
  file_type       text,
  description     text,
  created_at      timestamp without time zone,
  updated_at      timestamp without time zone
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw_stocks.dim_edgar_file_types
  OWNER TO postgres;
