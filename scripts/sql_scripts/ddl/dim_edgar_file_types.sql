-- Table: dw.fact_fred_releases

-- DROP TABLE dw.fact_fred_releases;

CREATE TABLE IF NOT EXISTS dw_stocks.dim_edgar_file_types
(
  file_type       text,
  description     text,
  dw_updated_at   timestamp without time zone,
  dw_created_at   timestamp without time zone
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw_stocks.dim_edgar_file_types
  OWNER TO postgres;
