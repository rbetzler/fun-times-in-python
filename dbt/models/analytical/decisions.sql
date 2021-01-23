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
, base as (
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
    , b.risk_neutral_probability - ((1 - b.risk_neutral_probability) / o.price / o.strike) as kelly_criterion
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
    and o.days_to_expiration > 0
)
, bools as (
  select *
    , thirty_day_low_prediction > strike as has_strike_below_30_day_predicted_low
    , days_to_expiration between 20 and 60 as has_sufficient_days_to_expiration
    , potential_annual_return > .2 as has_sufficient_potential_return
    , oom_percent > .15 as is_sufficiently_oom
    , risk_neutral_probability between .1 and .5 as has_sufficient_probability
    , pe_ratio between 10 and 40 as has_sufficient_pe
    , market_capitalization > 10 as has_sufficient_market_cap
    , theta - theta_half < theta_half - theta_quarter as has_early_theta_decay
    , avg_open_20 between avg_open_10 and avg_open_30 as has_30_day_trend
    , avg_open_60 between avg_open_10 and avg_open_90 as has_90_day_trend
    , avg_open_120 between avg_open_10 and avg_open_240 as has_240_day_trend
  from base
)
, tradables as (
  select *
    , coalesce(has_strike_below_30_day_predicted_low
      and has_sufficient_days_to_expiration
      and has_sufficient_potential_return
      and is_sufficiently_oom
      and has_sufficient_probability
      and has_sufficient_pe
      and has_sufficient_market_cap
      and has_early_theta_decay
      and has_30_day_trend
      and has_90_day_trend
      and has_240_day_trend, false) as is_tradable
  from bools
)
, row_numbers as (
  select *
    , row_number() over (w order by risk_neutral_probability desc) as rn_probability
    , row_number() over (w order by potential_annual_return desc) as rn_return
    , row_number() over (w order by oom_percent desc) as rn_oom
    , row_number() over (w order by theta) as rn_theta
  from tradables
  window w as (partition by symbol, market_datetime, days_to_expiration, is_tradable)
)
, final as (
  select
      model_id
    , market_datetime
    , symbol
    , thirty_day_low_prediction
    , stock_closing_price
    , is_call
    , days_to_expiration
    , strike
    , option_price
    , potential_annual_return
    , oom_percent
    , theta
    , theta_half
    , theta_quarter
    , theta_tenth
    , implied_volatility
    , risk_neutral_probability
    , kelly_criterion
    , pe_ratio
    , dividend_amount
    , dividend_date
    , market_capitalization
    , eps_trailing_twelve_months
    , quick_ratio
    , current_ratio
    , total_debt_to_equity
    , avg_open_10
    , avg_open_20
    , avg_open_30
    , avg_open_60
    , avg_open_90
    , avg_open_120
    , avg_open_180
    , avg_open_240
    , has_strike_below_30_day_predicted_low
    , has_sufficient_days_to_expiration
    , has_sufficient_potential_return
    , is_sufficiently_oom
    , has_sufficient_probability
    , has_sufficient_pe
    , has_sufficient_market_cap
    , has_early_theta_decay
    , has_30_day_trend
    , has_90_day_trend
    , has_240_day_trend
    , is_tradable and 1 = row_number() over (partition by symbol, market_datetime, days_to_expiration, is_tradable order by rn_probability + rn_return + rn_oom + rn_theta) as should_place_trade
  from row_numbers
)
select *
from final
