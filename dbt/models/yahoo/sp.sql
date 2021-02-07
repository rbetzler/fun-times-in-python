{{
  config(
    materialized='table',
    post_hook='''
      create index if not exists {{ this.name }}_market_datetime_idx on {{ this }} (market_datetime);
      '''
  )
}}

with
final as (
  select *
    , row_number() over (partition by market_datetime order by file_datetime desc) as rn
  from {{ source('yahoo', 'sp_index') }}
  where file_datetime >= current_date - 7
)
select
    market_datetime
  , open
  , high
  , low
  , close
  , volume
from final
where rn = 1
