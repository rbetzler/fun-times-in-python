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
stocks as (
  select *
    , dense_rank() over (partition by symbol order by market_datetime desc) as dr
  from {{ ref('stocks') }} as s
)
, base as (
  select
      s.market_datetime
    , s.symbol
    , s.close as stock_closing_price
    , o.is_call
    , o.days_to_expiration
    , o.strike
    , o.price as option_price
    , o.bid
    , o.ask
    , o.bid_ask_spread
    , o.open_interest
    , case
      when o.is_call then (o.strike - s.close) / s.close
      else (s.close - o.strike) / s.close
      end as oom_percent
    , b.delta
    , abs(b.delta) as abs_delta
    , b.gamma
    , b.vega
    , b.theta
    , b.theta / nullif(b.gamma, 0) as theta_gamma_offset
    , b.theta / nullif(b.vega, 0) as theta_vega_offset
    , b.implied_volatility
    , b.risk_neutral_probability
    , b.risk_neutral_probability - nullif((1 - b.risk_neutral_probability) / o.price / o.strike, 0) as kelly_criterion
    , t.avg_open_20 between t.avg_open_10 and t.avg_open_30 as has_short_downtrend
    , t.avg_open_120 between t.avg_open_60 and t.avg_open_240 as has_long_downtrend
    , t.avg_open_20 between t.avg_open_30 and t.avg_open_10 as has_short_uptrend
    , t.avg_open_120 between t.avg_open_240 and t.avg_open_60 as has_long_uptrend
  from stocks as s
  inner join {{ ref('options') }} as o
    on  s.symbol = o.symbol
    and s.market_datetime = o.market_datetime
  left join {{ ref('black_scholes') }} as b
    on  o.symbol = b.symbol
    and o.market_datetime = b.market_datetime
    and o.strike = b.strike
    and o.days_to_expiration = b.days_to_maturity
    and o.is_call = b.is_call
  inner join {{ ref('fundamentals') }} as f
    on  s.symbol = f.symbol
    and s.market_datetime = f.market_datetime
  left join {{ ref('technicals') }} as t
    on  s.symbol = t.symbol
    and s.market_datetime = t.market_datetime
  where s.dr = 1
    and s.close > 0
    and o.days_to_expiration between 10 and 60
    and o.price between .3 and 3
    and o.strike > 0
    and o.bid_ask_spread / nullif(o.bid, 0) < .2
    and o.open_interest > 0
    and f.pe_ratio between 0 and 30
    and f.market_capitalization > 10000
)
, windowed as (
  select *
    , percent_rank() over (w order by risk_neutral_probability) as pr_probability
    , percent_rank() over (w order by oom_percent) as pr_oom
    , percent_rank() over (w order by abs_delta) as pr_delta
    , percent_rank() over (w order by theta desc) as pr_theta
    , percent_rank() over (w order by theta_gamma_offset desc) as pr_theta_gamma_offset
    , percent_rank() over (w order by theta_vega_offset desc) as pr_theta_vega_offset
    , percent_rank() over (w order by implied_volatility) as pr_implied_volatility
  from base
  window w as (partition by symbol, market_datetime, days_to_expiration)
)
, final as (
  select
      market_datetime
    , symbol
    , stock_closing_price
    , is_call
    , days_to_expiration
    , strike
    , option_price
    , bid
    , ask
    , bid_ask_spread
    , open_interest
    , oom_percent
    , delta
    , gamma
    , vega
    , theta
    , theta_gamma_offset
    , theta_vega_offset
    , implied_volatility
    , risk_neutral_probability
    , kelly_criterion
    , has_short_uptrend
    , has_long_uptrend
    , has_short_downtrend
    , has_long_downtrend
    , 1 = row_number() over (
        partition by symbol, market_datetime
        order by pr_probability + pr_oom + pr_delta + pr_theta + pr_theta_gamma_offset + pr_theta_vega_offset + pr_implied_volatility
      ) as should_place_trade
  from windowed
)
select *
from final
where should_place_trade
