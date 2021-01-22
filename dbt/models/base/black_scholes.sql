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
final as (
  select *
    , row_number() over (partition by symbol, market_datetime, strike, days_to_maturity, is_call order by file_datetime desc) as rn
  from {{ source('td', 'black_scholes') }}
  where file_datetime >= current_date - 7
)
select *
from final
where rn = 1
