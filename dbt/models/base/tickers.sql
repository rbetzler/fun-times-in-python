{{
  config(
    materialized='table',
  )
}}

select distinct
    ticker as symbol
  , lower(regexp_replace(sector, '\W+', '_', 'g')) as sector
  , lower(regexp_replace(industry, '\W+', '_', 'g')) as industry
from {{ source('nasdaq', 'listed_stocks') }}
where ticker !~ '[\^.~]'
  and character_length(ticker) between 1 and 4
order by 1
