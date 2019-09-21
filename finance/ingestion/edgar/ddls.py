FILE_TYPES = """
    CREATE TABLE IF NOT EXISTS edgar.file_types
    (
      file_type       text,
      description     text,
      file_datetime   timestamp without time zone,
      ingest_datetime timestamp without time zone
    );
    """
SIC_CIK_CODES = """
    CREATE TABLE IF NOT EXISTS edgar.sic_cik_codes
    (
      company_name      text,
      cik_code          text,
      sic_code          text,
      file_datetime     timestamp without time zone,
      ingest_datetime   timestamp without time zone
    );
"""
FILINGS = """
    CREATE TABLE IF NOT EXISTS edgar.filings
    (
        company_name        text,
        filing_type         text,
        cik_code            text,
        date                timestamp without time zone,
        url                 text,
        file_datetime       timestamp without time zone,
        ingest_datetime     timestamp without time zone
    );
"""
