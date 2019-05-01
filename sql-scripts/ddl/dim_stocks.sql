-- Table: dw.fact_fred_releases

-- DROP TABLE dw.fact_fred_releases;

CREATE TABLE dw_stocks.dim_stocks
(
  ticker            text,
  company           text,
  adr_tso           text,
  ipo_year          text,
  sector            text,
  industry          text,
  exchange          text,
  dw_created_at     timestamp without time zone,
  dw_updated_at     timestamp without time zone
)
WITH (
  OIDS=FALSE
);
ALTER TABLE dw_stocks.dim_stocks
  OWNER TO postgres;
