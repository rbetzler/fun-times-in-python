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
predictions as (
  select *
    , dense_rank() over (partition by model_id order by file_datetime desc, market_datetime desc) as dr
  from {{ source('dev', 'predictions') }}
  where model_id = 's1'
)
select
    p.model_id
  , p.market_datetime
  , p.symbol
  , p.denormalized_prediction as thirty_day_low_prediction
  , s.close as stock_closing_price
  , o.is_call
  , o.days_to_expiration
  , o.strike
  , o.price as option_price
  , (o.price / o.strike) * (360 / o.days_to_expiration) as potential_annual_return
  , (s.close - o.strike) / s.close as oom_percent
  , b.theta
  , b.theta_half
  , b.theta_quarter
  , b.theta_tenth
  , b.implied_volatility
  , b.risk_neutral_probability
  , f.pe_ratio
  , f.dividend_amount
  , f.dividend_date
  , f.market_capitalization
  , f.eps_ttm as eps_trailing_twelve_months
  , f.quick_ratio
  , f.current_ratio
  , f.total_debt_to_equity
  , t.avg_open_10
  , t.avg_open_20
  , t.avg_open_30
  , t.avg_open_60
  , t.avg_open_90
  , t.avg_open_120
  , t.avg_open_180
  , t.avg_open_240
  -- , d.kelly_criterion
from predictions as p
left join {{ ref('stocks') }} as s
  on  p.symbol = s.symbol
  and p.market_datetime = s.market_datetime
left join {{ ref('options') }} as o
  on  p.symbol = o.symbol
  and p.market_datetime = o.market_datetime
  and not o.is_call
left join {{ ref('black_scholes') }} as b
  on  o.symbol = b.symbol
  and o.market_datetime = b.market_datetime
  and o.strike = b.strike
  and o.days_to_expiration = b.days_to_maturity
  and not b.is_call
left join {{ ref('fundamentals') }} as f
  on  p.symbol = f.symbol
  and p.market_datetime = f.market_datetime
left join {{ ref('technicals') }} as t
  on  p.symbol = t.symbol
  and p.market_datetime = t.market_datetime
where p.dr = 1
order by
    p.market_datetime
  , p.symbol
  , o.days_to_expiration
  , o.strike
