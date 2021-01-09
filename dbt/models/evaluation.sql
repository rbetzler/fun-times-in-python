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
p as (
  select *
    , row_number() over (partition by model_id, symbol, market_datetime order by file_datetime desc) as rn
  from {{ source('dev', 'predictions') }}
)
, base as (
  select
      p.model_id
    , t.market_datetime
    , t.symbol
    , t.denormalized_target as target
    , p.denormalized_prediction as prediction
    , p.denormalized_prediction - t.denormalized_target as error
    , abs(p.denormalized_prediction - t.denormalized_target) as abs_error
    , t.mean_deviation_10
    , t.mean_deviation_30
    , t.mean_deviation_60
    , t.mean_deviation_90
    , t.max_deviation_10
    , t.max_deviation_30
    , t.max_deviation_60
    , t.max_deviation_90
  from {{ ref('training') }} as t
  inner join p
    on  p.symbol = t.symbol
    and p.market_datetime = t.market_datetime
    and p.rn = 1
)
select
    model_id
  , market_datetime
  , symbol
  , target
  , prediction
  , error
  , abs_error
  , avg(abs_error) over (w rows between 10 preceding and current row) as mean_error_10
  , avg(abs_error) over (w rows between 30 preceding and current row) as mean_error_30
  , avg(abs_error) over (w rows between 60 preceding and current row) as mean_error_60
  , avg(abs_error) over (w rows between 90 preceding and current row) as mean_error_90
  , max(abs_error) over (w rows between 10 preceding and current row) as max_error_10
  , max(abs_error) over (w rows between 30 preceding and current row) as max_error_30
  , max(abs_error) over (w rows between 60 preceding and current row) as max_error_60
  , max(abs_error) over (w rows between 90 preceding and current row) as max_error_90
  , mean_deviation_10
  , mean_deviation_30
  , mean_deviation_60
  , mean_deviation_90
  , max_deviation_10
  , max_deviation_30
  , max_deviation_60
  , max_deviation_90
from base
window w as (partition by model_id, symbol order by market_datetime)
