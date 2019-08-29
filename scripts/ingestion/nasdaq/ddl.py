QUERY = """
    CREATE TABLE IF NOT EXISTS nasdaq.listed_stocks
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
    );
    """
