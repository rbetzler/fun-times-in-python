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
JOBS = """
  CREATE TABLE IF NOT EXISTS fred.jobs
  (
    series_id       text,
    series_name     text,
    category        text,
    is_active       boolean,
    dw_created_at   timestamp without time zone
  );
  INSERT INTO fred.jobs(series_id, series_name, category, is_active, dw_created_at) VALUES 
  ('FPCPITOTLZGWLD', 'inflation_world', 'inflation', True, NOW()),
  ('FPCPITOTLZGUSA', 'inflation_usa', 'inflation', True, NOW()),
  ('FPCPITOTLZGAUS', 'australia', 'inflation', True, NOW()),
  ('FPCPITOTLZGAUT', 'austria', 'inflation', True, NOW()),
  ('FPCPITOTLZGBEL', 'belgium', 'inflation', True, NOW()),
  ('FPCPITOTLZGBRA', 'brazil', 'inflation', True, NOW()),
  ('FPCPITOTLZGCAN', 'canada', 'inflation', True, NOW()),
  ('FPCPITOTLZGCHN', 'china', 'inflation', True, NOW()),
  ('FPCPITOTLZGEAP', 'developing_countries_in_east_asia_and_pacific', 'inflation', True, NOW()),
  ('FPCPITOTLZGECA', 'developing_countries_in_europe_and_central_asia', 'inflation', True, NOW()),
  ('FPCPITOTLZGLAC', 'developing_countries_in_latin_america_and_caribbean', 'inflation', True, NOW()),
  ('FPCPITOTLZGMNA', 'developing_countries_in_middle_east_and_north africa', 'inflation', True, NOW()),
  ('FPCPITOTLZGSSA', 'developing_countries_in_sub_saharan_africa', 'inflation', True, NOW()),
  ('FPCPITOTLZGFRA', 'france', 'inflation', True, NOW()),
  ('FPCPITOTLZGDEU', 'germany', 'inflation', True, NOW()),
  ('FPCPITOTLZGGRC', 'greece', 'inflation', True, NOW()),
  ('FPCPITOTLZGIND', 'india', 'inflation', True, NOW()),
  ('FPCPITOTLZGIDN', 'indonesia', 'inflation', True, NOW()),
  ('FPCPITOTLZGITA', 'italy', 'inflation', True, NOW()),
  ('FPCPITOTLZGJPN', 'japan', 'inflation', True, NOW()),
  ('FPCPITOTLZGLMY', 'low_and_middle_income_countries', 'inflation', True, NOW()),
  ('FPCPITOTLZGMEX', 'mexico', 'inflation', True, NOW()),
  ('FPCPITOTLZGOED', 'oecd_members', 'inflation', True, NOW()),
  ('FPCPITOTLZGPOL', 'poland', 'inflation', True, NOW()),
  ('FPCPITOTLZGSAU', 'saudi_arabia', 'inflation', True, NOW()),
  ('FPCPITOTLZGZAF', 'south_africa', 'inflation', True, NOW()),
  ('FPCPITOTLZGESP', 'spain', 'inflation', True, NOW()),
  ('FPCPITOTLZGCHE', 'switzerland', 'inflation', True, NOW()),
  ('FPCPITOTLZGTUR', 'turkey', 'inflation', True, NOW()),
  ('FPCPITOTLZGARB', 'the_arab_world', 'inflation', True, NOW()),
  ('FPCPITOTLZGEMU', 'the_euro_area', 'inflation', True, NOW()),
  ('FPCPITOTLZGEUU', 'the_european_union', 'inflation', True, NOW()),
  ('FPCPITOTLZGKOR', 'the_republic_of_korea', 'inflation', True, NOW()),
  ('FPCPITOTLZGRUS', 'the_russian_federation', 'inflation', True, NOW()),
  ('FPCPITOTLZGGBR', 'the_united_kingdom', 'inflation', True, NOW()),
  ('FPCPITOTLZGUSA', 'the_united_states', 'inflation', True, NOW()),
  ('FPCPITOTLZGWLD', 'the_world', 'inflation', True, NOW()),
  ('FPCPITOTLZGLCN', 'latin_america_and_caribbean', 'inflation', True, NOW());
  """