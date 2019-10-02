VALIDATION = """
    CREATE TABLE IF NOT EXISTS validation.benchmarks
    (
      market_datetime   timestamp without time zone,
      model_type        text,
      account_value     numeric(20,6),
      ingest_datetime   timestamp without time zone
    );
    """