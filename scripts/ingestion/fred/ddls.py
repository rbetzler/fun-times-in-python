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
      id                            text,
      realtime_start                timestamp without time zone,
      realtime_end                  timestamp without time zone,
      title                         text,
      observation_start             timestamp without time zone,
      observation_end               timestamp without time zone,
      frequency                     text,
      frequency_short               text,
      units                         text,
      units_short                   text,
      seasonal_adjustment           text,
      seasonal_adjustment_short     text,
      last_updated                  text,
      popularity                    text,
      group_popularity              text,
      notes                         text,
      dw_created_at                 timestamp without time zone
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
SERIES_SEARCHES = """
  CREATE TABLE IF NOT EXISTS fred.series_searches
  (
    search          text,
    is_active       boolean,
    dw_created_at   timestamp without time zone
  );
   INSERT INTO fred.series_searches(search, is_active, dw_created_at) VALUES 
  ('inflation', True, NOW()),
  ('gdp', True, NOW()),
  ('treasury', True, NOW());
  """
JOBS = f"""
  CREATE TABLE IF NOT EXISTS fred.jobs
  (
    series_id       text,
    series_name     text,
    category        text,
    is_active       boolean,
    dw_created_at   timestamp without time zone
  );
  INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at)
    with series as (
    select *
        , lower(title) as lower_title
        , trim(lower(regexp_replace(
            substring(title, position('for' in title)+4), 
            '[\s+\-]', '_', 'g'))) as subtitle
    from fred.series
    )
    select 
        id as series_id
        , subtitle as series_name
        , 'inflation' as category
        , true as is_active
        , NOW() as dw_created_at
    from series
    where lower(title) like '%inflation%'
    and not lower(title) like '%discontinued%'
    and not lower(title) like '%treasury%'
    and left(title, 30) = 'Inflation, consumer prices for'
    and (lower_title like '%australia%'
        or lower_title like '%germany%'
        or lower_title like '%france%'
        or lower_title like '%spain%'
        or lower_title like '%greece%'
        or lower_title like '%italy%'
        or lower_title like '%euro%'
        or lower_title like '%united kingdom%'
        or lower_title like '%united states%'
        or lower_title like '%canada%'
        or lower_title like '%mexico%'
        or lower_title like '%turkey%'
        or lower_title like '%poland%'
        or lower_title like '%brazil%'
        or lower_title like '%republic of korea%'
        or lower_title like '%china%'
        or lower_title like '%india%'
        or lower_title like '%russian federation%'
        or lower_title like '%countries%'
        or lower_title like '%oecd%')
    order by observation_start, title;
  """
