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
    order by symbol
    )
, lagged as (
    select
          t.symbol
        , t.sector
        , t.industry
        , s.market_datetime
        , min(s.open) over (partition by s.symbol order by s.market_datetime rows between 1 following and 31 following) as scaled_target
        , s.open
        , lag(s.open,  1) over w as open_1
        , lag(s.open,  2) over w as open_2
        , lag(s.open,  3) over w as open_3
        , lag(s.open,  4) over w as open_4
        , lag(s.open,  5) over w as open_5
        , lag(s.open,  6) over w as open_6
        , lag(s.open,  7) over w as open_7
        , lag(s.open,  8) over w as open_8
        , lag(s.open,  9) over w as open_9
        , lag(s.open, 10) over w as open_10
        , lag(s.open, 11) over w as open_11
        , lag(s.open, 12) over w as open_12
        , lag(s.open, 13) over w as open_13
        , lag(s.open, 14) over w as open_14
        , lag(s.open, 15) over w as open_15
        , lag(s.open, 16) over w as open_16
        , lag(s.open, 17) over w as open_17
        , lag(s.open, 18) over w as open_18
        , lag(s.open, 19) over w as open_19
        , lag(s.open, 20) over w as open_20
        , lag(s.open, 21) over w as open_21
        , lag(s.open, 22) over w as open_22
        , lag(s.open, 23) over w as open_23
        , lag(s.open, 24) over w as open_24
        , lag(s.open, 25) over w as open_25
        , lag(s.open, 26) over w as open_26
        , lag(s.open, 27) over w as open_27
        , lag(s.open, 28) over w as open_28
        , lag(s.open, 29) over w as open_29
        , lag(s.open, 30) over w as open_30
        , avg(s.open) over (w rows between 30 preceding and 1 preceding) as avg_open_30
        , avg(s.open) over (w rows between 60 preceding and 1 preceding) as avg_open_60
        , avg(s.open) over (w rows between 90 preceding and 1 preceding) as avg_open_90
        , min(s.open) over (w rows between 30 preceding and 1 preceding) as min_open_30
        , min(s.open) over (w rows between 60 preceding and 1 preceding) as min_open_60
        , min(s.open) over (w rows between 90 preceding and 1 preceding) as min_open_90
        , max(s.open) over (w rows between 30 preceding and 1 preceding) as max_open_30
        , max(s.open) over (w rows between 60 preceding and 1 preceding) as max_open_60
        , max(s.open) over (w rows between 90 preceding and 1 preceding) as max_open_90
    from {{ ref('stocks') }} as s
    inner join tickers as t
        on t.symbol = s.symbol
    where s.market_datetime > '2016-01-01'
    window w as (partition by s.symbol order by s.market_datetime)
    )
select
      symbol
    , sector
    , industry
    , market_datetime
    , scaled_target
    , 1 - (scaled_target / nullif(open, 0)) as target
    , open
    , 1 - (open_1 / nullif(open_2, 0)) as open_1
    , 1 - (open_2 / nullif(open_3, 0)) as open_2
    , 1 - (open_3 / nullif(open_4, 0)) as open_3
    , 1 - (open_4 / nullif(open_5, 0)) as open_4
    , 1 - (open_5 / nullif(open_6, 0)) as open_5
    , 1 - (open_6 / nullif(open_7, 0)) as open_6
    , 1 - (open_7 / nullif(open_8, 0)) as open_7
    , 1 - (open_8 / nullif(open_9, 0)) as open_8
    , 1 - (open_9 / nullif(open_10, 0)) as open_9
    , 1 - (open_10 / nullif(open_11, 0)) as open_10
    , 1 - (open_11 / nullif(open_12, 0)) as open_11
    , 1 - (open_12 / nullif(open_13, 0)) as open_12
    , 1 - (open_13 / nullif(open_14, 0)) as open_13
    , 1 - (open_14 / nullif(open_15, 0)) as open_14
    , 1 - (open_15 / nullif(open_16, 0)) as open_15
    , 1 - (open_16 / nullif(open_17, 0)) as open_16
    , 1 - (open_17 / nullif(open_18, 0)) as open_17
    , 1 - (open_18 / nullif(open_19, 0)) as open_18
    , 1 - (open_19 / nullif(open_20, 0)) as open_19
    , 1 - (open_20 / nullif(open_21, 0)) as open_20
    , 1 - (open_21 / nullif(open_22, 0)) as open_21
    , 1 - (open_22 / nullif(open_23, 0)) as open_22
    , 1 - (open_23 / nullif(open_24, 0)) as open_23
    , 1 - (open_24 / nullif(open_25, 0)) as open_24
    , 1 - (open_25 / nullif(open_26, 0)) as open_25
    , 1 - (open_26 / nullif(open_27, 0)) as open_26
    , 1 - (open_27 / nullif(open_28, 0)) as open_27
    , 1 - (open_28 / nullif(open_29, 0)) as open_28
    , 1 - (open_29 / nullif(open_30, 0)) as open_29
    , 1 - (avg_open_30 / nullif(avg_open_60, 0)) as avg_open_30_over_60
    , 1 - (avg_open_60 / nullif(avg_open_90, 0)) as avg_open_60_over_90
    , 1 - (min_open_30 / nullif(max_open_30, 0)) as min_over_max_30
    , 1 - (min_open_60 / nullif(max_open_60, 0)) as min_over_max_60
    , 1 - (min_open_90 / nullif(max_open_90, 0)) as min_over_max_90
    , 1 - (min_open_60 / nullif(min_open_30, 0)) as min_open_60_over_30
    , 1 - (min_open_90 / nullif(min_open_60, 0)) as min_open_90_over_60
from lagged
where market_datetime > '2017-01-01'
  and open_30 is not null
order by market_datetime, symbol
