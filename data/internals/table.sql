create table if not exists audit.ingest_datetimes (
  id                serial primary key,
  schema_name       text,
  table_name        text,
  job_name          text,
  ingest_datetime   timestamp without time zone
);
