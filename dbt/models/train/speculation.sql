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
tickers as (
  select distinct symbol, sector, industry
  from {{ ref('tickers') }}
)
, base as (
  select
      t.symbol
    , t.sector
    , t.industry
    , s.market_datetime
    , s.open
    , s.high
    , s.low
    , s.close
    , (s.high - s.low) / nullif(s.high, 0) as daily_range
    , (s.close - s.open) / nullif(s.open, 0) as intraday_performance
    , s.open / nullif(lag(s.close, 1) over (w), 0) as open_over_prior_close
    , s.open / nullif(max(s.open) over (w rows between 10 preceding and current row), 0) as open_over_10_day_max
    , s.open / nullif(max(s.open) over (w rows between 30 preceding and current row), 0) as open_over_30_day_max
    , s.open / nullif(max(s.open) over (w rows between 60 preceding and current row), 0) as open_over_60_day_max
    , s.open / nullif(max(s.open) over (w rows between 90 preceding and current row), 0) as open_over_90_day_max
    , s.volume / nullif(max(s.volume) over (w rows between 10 preceding and current row), 0) as volume_over_10_day_max
    , s.volume / nullif(max(s.volume) over (w rows between 30 preceding and current row), 0) as volume_over_30_day_max
    , s.volume / nullif(max(s.volume) over (w rows between 60 preceding and current row), 0) as volume_over_60_day_max
    , s.volume / nullif(max(s.volume) over (w rows between 90 preceding and current row), 0) as volume_over_90_day_max
    , min(s.high) over (w rows between 1 following and 10 following) as scaled_target_min
    , max(s.high) over (w rows between 1 following and 10 following) as scaled_target_max
  from {{ ref('stocks') }} as s
  inner join tickers as t
    on  s.symbol = t.symbol
  where s.market_datetime > '2018-01-01'
  window w as (partition by s.symbol order by s.market_datetime)
)
, windowed as (
  select
      market_datetime
    , symbol
    , sector
    , industry
    , close
    , scaled_target_min
    , scaled_target_max
    , scaled_target_min / close as target_min
    , scaled_target_max / close as target_max
    , coalesce(lag(daily_range, 1) over (w), 0) as daily_range_1
    , coalesce(lag(daily_range, 2) over (w), 0) as daily_range_2
    , coalesce(lag(daily_range, 3) over (w), 0) as daily_range_3
    , coalesce(lag(daily_range, 5) over (w), 0) as daily_range_5
    , coalesce(lag(daily_range, 10) over (w), 0) as daily_range_10
    , coalesce(lag(daily_range, 20) over (w), 0) as daily_range_20
    , coalesce(lag(daily_range, 30) over (w), 0) as daily_range_30
    , coalesce(lag(intraday_performance, 1) over (w), 0) as intraday_performance_1
    , coalesce(lag(intraday_performance, 2) over (w), 0) as intraday_performance_2
    , coalesce(lag(intraday_performance, 3) over (w), 0) as intraday_performance_3
    , coalesce(lag(intraday_performance, 5) over (w), 0) as intraday_performance_5
    , coalesce(lag(intraday_performance, 10) over (w), 0) as intraday_performance_10
    , coalesce(lag(intraday_performance, 20) over (w), 0) as intraday_performance_20
    , coalesce(lag(intraday_performance, 30) over (w), 0) as intraday_performance_30
    , coalesce(lag(open_over_prior_close, 1) over (w), 0) as open_over_prior_close_1
    , coalesce(lag(open_over_prior_close, 2) over (w), 0) as open_over_prior_close_2
    , coalesce(lag(open_over_prior_close, 3) over (w), 0) as open_over_prior_close_3
    , coalesce(lag(open_over_prior_close, 5) over (w), 0) as open_over_prior_close_5
    , coalesce(lag(open_over_prior_close, 10) over (w), 0) as open_over_prior_close_10
    , coalesce(lag(open_over_prior_close, 20) over (w), 0) as open_over_prior_close_20
    , coalesce(lag(open_over_prior_close, 30) over (w), 0) as open_over_prior_close_30
    , coalesce(lag(open_over_10_day_max, 1) over (w), 0) as open_over_10_day_max_1
    , coalesce(lag(open_over_10_day_max, 2) over (w), 0) as open_over_10_day_max_2
    , coalesce(lag(open_over_10_day_max, 3) over (w), 0) as open_over_10_day_max_3
    , coalesce(lag(open_over_10_day_max, 5) over (w), 0) as open_over_10_day_max_5
    , coalesce(lag(open_over_10_day_max, 10) over (w), 0) as open_over_10_day_max_10
    , coalesce(lag(open_over_10_day_max, 20) over (w), 0) as open_over_10_day_max_20
    , coalesce(lag(open_over_10_day_max, 30) over (w), 0) as open_over_10_day_max_30
    , coalesce(lag(open_over_30_day_max, 1) over (w), 0) as open_over_30_day_max_1
    , coalesce(lag(open_over_30_day_max, 2) over (w), 0) as open_over_30_day_max_2
    , coalesce(lag(open_over_30_day_max, 3) over (w), 0) as open_over_30_day_max_3
    , coalesce(lag(open_over_30_day_max, 5) over (w), 0) as open_over_30_day_max_5
    , coalesce(lag(open_over_30_day_max, 10) over (w), 0) as open_over_30_day_max_10
    , coalesce(lag(open_over_30_day_max, 20) over (w), 0) as open_over_30_day_max_20
    , coalesce(lag(open_over_30_day_max, 30) over (w), 0) as open_over_30_day_max_30
    , coalesce(lag(open_over_60_day_max, 1) over (w), 0) as open_over_60_day_max_1
    , coalesce(lag(open_over_60_day_max, 2) over (w), 0) as open_over_60_day_max_2
    , coalesce(lag(open_over_60_day_max, 3) over (w), 0) as open_over_60_day_max_3
    , coalesce(lag(open_over_60_day_max, 5) over (w), 0) as open_over_60_day_max_5
    , coalesce(lag(open_over_60_day_max, 10) over (w), 0) as open_over_60_day_max_10
    , coalesce(lag(open_over_60_day_max, 20) over (w), 0) as open_over_60_day_max_20
    , coalesce(lag(open_over_60_day_max, 30) over (w), 0) as open_over_60_day_max_30
    , coalesce(lag(open_over_90_day_max, 1) over (w), 0) as open_over_90_day_max_1
    , coalesce(lag(open_over_90_day_max, 2) over (w), 0) as open_over_90_day_max_2
    , coalesce(lag(open_over_90_day_max, 3) over (w), 0) as open_over_90_day_max_3
    , coalesce(lag(open_over_90_day_max, 5) over (w), 0) as open_over_90_day_max_5
    , coalesce(lag(open_over_90_day_max, 10) over (w), 0) as open_over_90_day_max_10
    , coalesce(lag(open_over_90_day_max, 20) over (w), 0) as open_over_90_day_max_20
    , coalesce(lag(open_over_90_day_max, 30) over (w), 0) as open_over_90_day_max_30
    , coalesce(lag(volume_over_10_day_max, 1) over (w), 0) as volume_over_10_day_max_1
    , coalesce(lag(volume_over_10_day_max, 2) over (w), 0) as volume_over_10_day_max_2
    , coalesce(lag(volume_over_10_day_max, 3) over (w), 0) as volume_over_10_day_max_3
    , coalesce(lag(volume_over_10_day_max, 5) over (w), 0) as volume_over_10_day_max_5
    , coalesce(lag(volume_over_10_day_max, 10) over (w), 0) as volume_over_10_day_max_10
    , coalesce(lag(volume_over_10_day_max, 20) over (w), 0) as volume_over_10_day_max_20
    , coalesce(lag(volume_over_10_day_max, 30) over (w), 0) as volume_over_10_day_max_30
    , coalesce(lag(volume_over_30_day_max, 1) over (w), 0) as volume_over_30_day_max_1
    , coalesce(lag(volume_over_30_day_max, 2) over (w), 0) as volume_over_30_day_max_2
    , coalesce(lag(volume_over_30_day_max, 3) over (w), 0) as volume_over_30_day_max_3
    , coalesce(lag(volume_over_30_day_max, 5) over (w), 0) as volume_over_30_day_max_5
    , coalesce(lag(volume_over_30_day_max, 10) over (w), 0) as volume_over_30_day_max_10
    , coalesce(lag(volume_over_30_day_max, 20) over (w), 0) as volume_over_30_day_max_20
    , coalesce(lag(volume_over_30_day_max, 30) over (w), 0) as volume_over_30_day_max_30
    , coalesce(lag(volume_over_60_day_max, 1) over (w), 0) as volume_over_60_day_max_1
    , coalesce(lag(volume_over_60_day_max, 2) over (w), 0) as volume_over_60_day_max_2
    , coalesce(lag(volume_over_60_day_max, 3) over (w), 0) as volume_over_60_day_max_3
    , coalesce(lag(volume_over_60_day_max, 5) over (w), 0) as volume_over_60_day_max_5
    , coalesce(lag(volume_over_60_day_max, 10) over (w), 0) as volume_over_60_day_max_10
    , coalesce(lag(volume_over_60_day_max, 20) over (w), 0) as volume_over_60_day_max_20
    , coalesce(lag(volume_over_60_day_max, 30) over (w), 0) as volume_over_60_day_max_30
    , coalesce(lag(volume_over_90_day_max, 1) over (w), 0) as volume_over_90_day_max_1
    , coalesce(lag(volume_over_90_day_max, 2) over (w), 0) as volume_over_90_day_max_2
    , coalesce(lag(volume_over_90_day_max, 3) over (w), 0) as volume_over_90_day_max_3
    , coalesce(lag(volume_over_90_day_max, 5) over (w), 0) as volume_over_90_day_max_5
    , coalesce(lag(volume_over_90_day_max, 10) over (w), 0) as volume_over_90_day_max_10
    , coalesce(lag(volume_over_90_day_max, 20) over (w), 0) as volume_over_90_day_max_20
    , coalesce(lag(volume_over_90_day_max, 30) over (w), 0) as volume_over_90_day_max_30
  from base
  where open > 0
    and high > 0
    and low > 0
    and close > 0
  window w as (partition by symbol order by market_datetime)
)
select *
from windowed
where market_datetime > '2019-01-01'
order by market_datetime, symbol
