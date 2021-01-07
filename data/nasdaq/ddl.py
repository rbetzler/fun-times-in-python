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
      file_datetime     timestamp without time zone,
      ingest_datetime   timestamp without time zone
    );
    """
