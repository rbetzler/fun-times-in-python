QUERY = """
    CREATE TABLE IF NOT EXISTS audit.ingest_datetimes
    (
      schema_name       text,
      table_name        text,
      job_name          text,
      ingest_datetime   timestamp without time zone
    );
    """
