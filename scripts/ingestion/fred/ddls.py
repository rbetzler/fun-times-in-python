RELEASES = """
  CREATE TABLE IF NOT EXISTS fred.releases
  (
    release_id      numeric,
    link            text,
    name            text,
    press_release   text,
    realtime_end    date,
    realtime_start  date,
    source_id       numeric,
    dw_created_at   timestamp without time zone,
    dw_updated_at   timestamp without time zone
  );
  """
SERIES = """
  CREATE TABLE IF NOT EXISTS fred.series
  (
    release_id              numeric,
    series_id               text,
    frequency_long          text,
    frequency               text,
    group_popularity        numeric,
    last_updated            timestamp without time zone,
    notes                   text,
    observation_end_date    date,
    observation_start_date  date,
    popularity              numeric,
    realtime_end            date,
    realtime_start          date,
    seasonal_adj_long       text,
    seasonal_adj            text,
    title                   text,
    units_long              text,
    units                   text,
    dw_created_at           timestamp without time zone,
    dw_updated_at           timestamp without time zone
  );
  """
SOURCES = """
  CREATE TABLE IF NOT EXISTS fred.sources
  (
    source_id         numeric,
    link              text,
    name              text,
    realtime_end      date,
    realtime_start    date,
    dw_created_at     timestamp without time zone,
    dw_updated_at     timestamp without time zone
  );
  """