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
partitioned as (
  select *
    , row_number() over(partition by symbol, market_datetime order by file_datetime desc) as rn
  from {{ source('td', 'equities_raw') }}
)
, final as (
  select
      symbol
    , open
    , high
    , low
    , close
    , volume
    , market_datetime_epoch
    , empty
    , market_datetime::date as market_datetime
    , file_datetime
    , ingest_datetime
  from partitioned
  where rn = 1
)
select *
from final
