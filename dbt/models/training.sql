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
    select symbol as ticker
    from {{ ref('stocks') }}
    where market_datetime between '2010-01-15' and '2020-11-10'
    group by 1
    having bool_or(market_datetime = ('2015-01-15'))
       and bool_or(market_datetime = ('2019-06-10'))
    order by symbol
    limit 75
    )
, lagged as (
    select
          s.symbol
        , s.market_datetime
        , min(s.open) over (partition by s.symbol order by s.market_datetime rows between 1 following and 31 following) as target
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
    from {{ ref('stocks') }} as s
    inner join tickers as t
        on t.ticker = s.symbol
    window w as (partition by s.symbol order by s.market_datetime)
    )
, summarized as (
    select *
        , least(
              open_1
            , open_2
            , open_3
            , open_4
            , open_5
            , open_6
            , open_7
            , open_8
            , open_9
            , open_10
            , open_11
            , open_12
            , open_13
            , open_14
            , open_15
            , open_16
            , open_17
            , open_18
            , open_19
            , open_20
            , open_21
            , open_22
            , open_23
            , open_24
            , open_25
            , open_26
            , open_27
            , open_28
            , open_29
            , open_30
          ) as normalization_min
        , greatest(
              open_1
            , open_2
            , open_3
            , open_4
            , open_5
            , open_6
            , open_7
            , open_8
            , open_9
            , open_10
            , open_11
            , open_12
            , open_13
            , open_14
            , open_15
            , open_16
            , open_17
            , open_18
            , open_19
            , open_20
            , open_21
            , open_22
            , open_23
            , open_24
            , open_25
            , open_26
            , open_27
            , open_28
            , open_29
            , open_30
          ) as normalization_max
    from lagged
    )
select
      symbol
    , market_datetime
    , (target - normalization_min) / (normalization_max - normalization_min) as target
    , target as denormalized_target
    , normalization_min
    , normalization_max
    , (open_1  - normalization_min) / (normalization_max - normalization_min) as open_1
    , (open_2  - normalization_min) / (normalization_max - normalization_min) as open_2
    , (open_3  - normalization_min) / (normalization_max - normalization_min) as open_3
    , (open_4  - normalization_min) / (normalization_max - normalization_min) as open_4
    , (open_5  - normalization_min) / (normalization_max - normalization_min) as open_5
    , (open_6  - normalization_min) / (normalization_max - normalization_min) as open_6
    , (open_7  - normalization_min) / (normalization_max - normalization_min) as open_7
    , (open_8  - normalization_min) / (normalization_max - normalization_min) as open_8
    , (open_9  - normalization_min) / (normalization_max - normalization_min) as open_9
    , (open_10 - normalization_min) / (normalization_max - normalization_min) as open_10
    , (open_11 - normalization_min) / (normalization_max - normalization_min) as open_11
    , (open_12 - normalization_min) / (normalization_max - normalization_min) as open_12
    , (open_13 - normalization_min) / (normalization_max - normalization_min) as open_13
    , (open_14 - normalization_min) / (normalization_max - normalization_min) as open_14
    , (open_15 - normalization_min) / (normalization_max - normalization_min) as open_15
    , (open_16 - normalization_min) / (normalization_max - normalization_min) as open_16
    , (open_17 - normalization_min) / (normalization_max - normalization_min) as open_17
    , (open_18 - normalization_min) / (normalization_max - normalization_min) as open_18
    , (open_19 - normalization_min) / (normalization_max - normalization_min) as open_19
    , (open_20 - normalization_min) / (normalization_max - normalization_min) as open_20
    , (open_21 - normalization_min) / (normalization_max - normalization_min) as open_21
    , (open_22 - normalization_min) / (normalization_max - normalization_min) as open_22
    , (open_23 - normalization_min) / (normalization_max - normalization_min) as open_23
    , (open_24 - normalization_min) / (normalization_max - normalization_min) as open_24
    , (open_25 - normalization_min) / (normalization_max - normalization_min) as open_25
    , (open_26 - normalization_min) / (normalization_max - normalization_min) as open_26
    , (open_27 - normalization_min) / (normalization_max - normalization_min) as open_27
    , (open_28 - normalization_min) / (normalization_max - normalization_min) as open_28
    , (open_29 - normalization_min) / (normalization_max - normalization_min) as open_29
    , (open_30 - normalization_min) / (normalization_max - normalization_min) as open_30
from summarized
where open_30 is not null
  and normalization_max <> normalization_min
order by market_datetime, symbol
