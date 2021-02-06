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
raw_predictions as (
  select *
    , row_number() over (partition by model_id, symbol, market_datetime order by file_datetime desc) as rn
  from {{ source('dev', 'predictions') }}
  where model_id in ('s3', 's4')
)
, predictions as (
  select *
  from raw_predictions
  where rn = 1
)
, base as (
  select
      l.model_id as low_model_id
    , h.model_id as high_model_id
    , t.market_datetime
    , t.symbol
    , i.sector
    , i.industry
    , t.close as closing_price
    , t.scaled_target_min as target_low
    , t.scaled_target_max as target_high
    , l.scaled_prediction as prediction_low
    , h.scaled_prediction as prediction_high
    , (l.scaled_prediction - t.scaled_target_min) / nullif(t.scaled_target_min, 0) as low_error
    , abs(l.scaled_prediction - t.scaled_target_min) / nullif(t.scaled_target_min, 0) as low_abs_error
    , l.scaled_prediction < t.scaled_target_min as is_loss_low
    , (h.scaled_prediction - t.scaled_target_max) / nullif(t.scaled_target_max, 0) as high_error
    , abs(h.scaled_prediction - t.scaled_target_max) / nullif(t.scaled_target_max, 0) as high_abs_error
    , h.scaled_prediction < t.scaled_target_max as is_loss_high
  from {{ ref('speculation') }} as t
  inner join predictions as l
    on  l.symbol = t.symbol
    and l.market_datetime = t.market_datetime
    and l.model_id = 's3'
  inner join predictions as h
    on  h.symbol = t.symbol
    and h.market_datetime = t.market_datetime
    and h.model_id = 's4'
  left join {{ ref('tickers') }} as i
    on  i.symbol = t.symbol
)
, final as (
  select
      low_model_id
    , high_model_id
    , market_datetime
    , symbol
    , sector
    , industry
    , closing_price
    , target_low
    , target_high
    , prediction_low
    , prediction_high
    , low_error
    , low_abs_error
    , high_error
    , high_abs_error
    , is_loss_low
    , is_loss_high
    , case when is_loss_low then 1 else 0 end as n_loss_low
    , case when is_loss_high then 1 else 0 end as n_loss_high
  from base
)
select *
from final
