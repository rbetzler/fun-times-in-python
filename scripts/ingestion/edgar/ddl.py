QUERY = """
    CREATE TABLE IF NOT EXISTS edgar.file_types
    (
      file_type       text,
      description     text,
      dw_updated_at   timestamp without time zone,
      dw_created_at   timestamp without time zone
    );
    """
