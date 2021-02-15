{{
  config(
    materialized='table',
  )
}}

with
base as (
  select
      ticker as symbol
    , lower(regexp_replace(sector, '\W+', '_', 'g')) as sector
    , lower(regexp_replace(industry, '\W+', '_', 'g')) as industry
    , row_number() over (partition by ticker order by file_datetime desc) as rn
  from {{ source('nasdaq', 'listed_stocks') }}
  where ticker !~ '[\^.~]'
    and character_length(ticker) between 1 and 4
)
select
    symbol
  , sector
  , industry
from base
where rn = 1
order by 1
