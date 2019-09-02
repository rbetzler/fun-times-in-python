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
  ('gross', True, NOW()),
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
  
  --inflation
  INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at)
    with series as (
    select *
        , lower(title) as lower_title
        , trim(lower(regexp_replace(
            substring(title, position('for' in title)+4), 
            '[\s+\-]', '_', 'g'))) as subtitle
    from fred.series
    )
    select distinct
        id as series_id
        , subtitle as series_name
        , 'inflation' as category
        , true as is_active
        , NOW() as dw_created_at
    from series
    where lower(title) like '%inflation%'
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
        or lower_title like '%oecd%');
        
    --central government debt
    INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at)
    with series as (
        select *
            , lower(title) as lower_title
            , trim(lower(regexp_replace(
                substring(title, position('for' in title)+4), 
                '[\s+\-]', '_', 'g'))) as subtitle
        from fred.series
        )
    select distinct
        id as series_id
        , subtitle as series_name
        , 'central_government_debt' as category
        , true as is_active
        , NOW() as dw_created_at
    from series
    where left(title, 45) = 'Central government debt, total (% of GDP) for'
    and not lower(title) like '%discontinued%'
    and (lower_title like '%germany%'
        or lower_title like '%greece%'
        or lower_title like '%india%'
        or lower_title like '%italy%'
        or lower_title like '%japan%'
        or lower_title like '%turkey%'
        or lower_title like '%united states%');
        
    --household debt to gdp
    INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at)
    with series as (
        select *
            , lower(title) as lower_title
            , trim(lower(regexp_replace(
                substring(title, position('for' in title)+4), 
            '[\s+\-]', '_', 'g'))) as subtitle
    from fred.series
    )
    select distinct
        id as series_id
        , subtitle as series_name
        , 'household_debt_to_gdp' as category
        , true as is_active
        , NOW() as dw_created_at
    from series
    where left(title, 25) = 'Household Debt to GDP for'
    and frequency = 'Quarterly'
    and (lower_title like '%australia%'
        or lower_title like '%germany%'
        or lower_title like '%italy%'
        or lower_title like '%united kingdom%'
        or lower_title like '%united states%'
        or lower_title like '%canada%'
        or lower_title like '%republic of korea%');

    --real gdp per capita
    INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at)
    with series as (
        select *
            , lower(title) as lower_title
            , trim(lower(regexp_replace(
                substring(title, position('for' in title)+4), 
            '[\s+\-]', '_', 'g'))) as subtitle
    from fred.series
    )
    select distinct
        id as series_id
        , subtitle as series_name
        , 'real_gdp_per_capita' as category
        , true as is_active
        , NOW() as dw_created_at
    from series
    where left(title, 27) = 'Constant GDP per capita for'
        and (lower_title like '%germany%'
            or lower_title like '%france%'
            or lower_title like '%spain%'
            or lower_title like '%italy%'
            or lower_title like '%united kingdom%'
            or lower_title like '%united states%'
            or lower_title like '%canada%'
            or lower_title like '%mexico%'
            or lower_title like '%turkey%'
            or lower_title like '%brazil%'
            or lower_title like '%republic of korea%'
            or lower_title like '%china%'
            or lower_title like '%india%'
            or lower_title like '%russian federation%'
            or lower_title like '%world%');
         
    --stock market capitalization to gdp
    INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at)   
    with series as (
        select *
            , lower(title) as lower_title
            , trim(lower(regexp_replace(
                substring(title, position('for' in title)+4), 
            '[\s+\-]', '_', 'g'))) as subtitle
    from fred.series
    )
    select distinct
        id as series_id
        , subtitle as series_name
        , 'stock_market_capitalization_to_gdp' as category
        , true as is_active
        , NOW() as dw_created_at
    from series
    where left(title, 38) = 'Stock Market Capitalization to GDP for'
        and (lower_title like '%germany%'
            or lower_title like '%united kingdom%'
            or lower_title like '%united states%'
            or lower_title like '%canada%'
            or lower_title like '%brazil%'
            or lower_title like '%republic of korea%'
            or lower_title like '%china%'
            or lower_title like '%india%'
            or lower_title like '%russian federation%');

    --central bank assets to gdp
    INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at)
    with series as (
            select *
                , lower(title) as lower_title
                , trim(lower(regexp_replace(
                    substring(title, position('for' in title)+4), 
                    '[\s+\-]', '_', 'g'))) as subtitle
            from fred.series
            )
        select distinct
            id as series_id
            , subtitle as series_name
            , 'central_bank_assets_to_gdp' as category
            , true as is_active
            , NOW() as dw_created_at
        from series
        where left(title, 30) = 'Central Bank Assets to GDP for'
        and (lower_title like '%china%'
            or lower_title like '%japan%'
            or lower_title like '%united states%');
  """
