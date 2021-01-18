{{
  config(
    materialized='table',
    post_hook='''
      create index if not exists {{ this.name }}_symbol_idx on {{ this }} (symbol);
      create index if not exists {{ this.name }}_market_datetime_idx on {{ this }} (market_datetime);
      '''
  )
}}

with
raw_quotes as (
  select
      case when extract(hour from trade_time_in_long_datetime) < 10 then trade_time_in_long_datetime::date - 1 else trade_time_in_long_datetime::date end as market_datetime
    , symbol
    , open_price as open
    , high_price as high
    , low_price as low
    , regular_market_last_price as close
    , total_volume as volume
    , file_datetime
  from {{ source('td', 'quotes_raw') }}
  )
, partitioned as (
  select *
    , row_number() over (partition by symbol, market_datetime order by file_datetime desc) as rn
  from raw_quotes
  )
select
    market_datetime
  , symbol
  , open
  , high
  , low
  , close
  , volume
  , file_datetime
from partitioned
where rn = 1
