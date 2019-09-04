FILE_TYPES = """
    CREATE TABLE IF NOT EXISTS edgar.file_types
    (
      file_type       text,
      description     text,
      dw_updated_at   timestamp without time zone,
      dw_created_at   timestamp without time zone
    );
    """
SIC_CIK_CODES = """
    CREATE TABLE IF NOT EXISTS edgar.sic_cik_codes
    (
      company_name      text,
      cik_code          text,
      sic_code          text,
      dw_created_at     timestamp without time zone
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
        dw_created_at       timestamp without time zone
    );
"""
