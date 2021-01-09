{{
  config(
    materialized='table',
  )
}}

select distinct
    ticker as symbol
  , sector
  , industry
from {{ source('nasdaq', 'listed_stocks') }}
where ticker !~ '[\^.~]'
  and character_length(ticker) between 1 and 4
order by 1
