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
base as (
  select
      symbol
    , market_datetime
    , avg(open) over (w rows between 9 preceding and current row) as avg_open_10
    , avg(open) over (w rows between 19 preceding and current row) as avg_open_20
    , avg(open) over (w rows between 29 preceding and current row) as avg_open_30
    , avg(open) over (w rows between 59 preceding and current row) as avg_open_60
    , avg(open) over (w rows between 89 preceding and current row) as avg_open_90
    , avg(open) over (w rows between 119 preceding and current row) as avg_open_120
    , avg(open) over (w rows between 179 preceding and current row) as avg_open_180
    , avg(open) over (w rows between 239 preceding and current row) as avg_open_240
    , min(low) over (w rows between 29 preceding and current row) as min_30
    , min(low) over (w rows between 89 preceding and current row) as min_90
    , min(low) over (w rows between 239 preceding and current row) as min_240
    , max(high) over (w rows between 29 preceding and current row) as max_30
    , max(high) over (w rows between 89 preceding and current row) as max_90
    , max(high) over (w rows between 239 preceding and current row) as max_240
  from {{ ref('stocks') }}
  where market_datetime > current_date - 500
  window w as (partition by symbol order by market_datetime)
)
, final as (
  select *
  from base
  where market_datetime > current_date - 10
)
select *
from final
