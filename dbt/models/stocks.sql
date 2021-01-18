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
equities as (
  select distinct
      market_datetime
    , symbol
    , open
    , high
    , low
    , close
    , volume
  from {{ ref('equities') }}
)
, latest_equities as (
  select
      symbol
    , max(market_datetime) as market_datetime
  from equities
  group by symbol
)
, quotes as (
  select distinct
      q.market_datetime
    , q.symbol
    , q.open
    , q.high
    , q.low
    , q.close
    , q.volume
  from {{ ref('quotes') }} as q
  left join latest_equities as l
    on q.symbol = l.symbol
  where (l.symbol is null and l.market_datetime is null)
      or l.market_datetime < q.market_datetime
)
, final as (
  select
      market_datetime
    , symbol
    , open
    , high
    , low
    , close
    , volume
  from quotes
  union
  select
      market_datetime
    , symbol
    , open
    , high
    , low
    , close
    , volume
  from equities
)
select *
from final
