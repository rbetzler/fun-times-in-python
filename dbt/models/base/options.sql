{{
  config(
    materialized='table',
    post_hook='''
      create index if not exists {{ this.name }}_symbol_idx on {{ this }} (symbol);
      create index if not exists {{ this.name }}_file_datetime_idx on {{ this }} (file_datetime);
      '''
  )
}}

with
raw as (
  select
      o.symbol
    , o.volatility
    , o.n_contracts
    , o.interest_rate
    , o.put_call = 'CALL' as is_call
    , (o.bid + o.ask) / 2 as price
    , o.bid
    , o.ask
    , o.last
    , o.mark
    , o.bid_size
    , o.ask_size
    , o.bid_ask_size
    , o.last_size
    , o.high_price
    , o.low_price
    , o.open_price
    , o.close_price
    , o.total_volume
    , o.trade_date
    , o.delta
    , o.gamma
    , o.theta
    , o.vega
    , o.rho
    , o.open_interest
    , o.time_value
    , o.theoretical_option_value
    , o.theoretical_volatility
    , o.strike_price
    , o.expiration_date
    , o.days_to_expiration
    , o.expiration_date_from_epoch
    , o.strike
    , o.strike_date
    , o.days_to_expiration_date
    , o.ingest_datetime
    , case when h.day_date is not null then o.file_datetime - interval '1 days' else o.file_datetime end as file_datetime
  from {{ source('td', 'options_raw') }} as o
  left join {{ ref('holidays') }} as h
    on  o.file_datetime::date = h.day_date
  where o.file_datetime > current_date - 7 and o.file_datetime < current_date + 1
)
, dated as (
  select *
    , case
      -- If Sunday, set to Friday
      when extract(isodow from file_datetime) = 7 then file_datetime::date - 2
      -- If Saturday, set to Friday
      when extract(isodow from file_datetime) = 6 then file_datetime::date - 1
      -- If after market close, leave as is
      when extract('hour' from file_datetime) >= 20 then file_datetime::date
      -- If before market open, set to prior day
      when extract(hour from file_datetime) < 13 or (extract('hour' from file_datetime) = 13 and extract('minute' from file_datetime) <= 30) then file_datetime::date - 1
      end as file_date
  from raw
)
, ranked as (
  select *
    , row_number() over(partition by file_date, symbol, is_call, strike, days_to_expiration order by ingest_datetime desc) as rn
  from dated
  where file_date is not null
)
, final as (
  select
      symbol
    , volatility
    , n_contracts
    , interest_rate
    , is_call
    , price
    , bid
    , ask
    , last
    , mark
    , bid_size
    , ask_size
    , bid_ask_size
    , last_size
    , high_price
    , low_price
    , open_price
    , close_price
    , total_volume
    , trade_date
    , delta
    , gamma
    , theta
    , vega
    , rho
    , open_interest
    , time_value
    , theoretical_option_value
    , theoretical_volatility
    , strike_price
    , expiration_date
    , days_to_expiration
    , expiration_date_from_epoch
    , strike
    , strike_date
    , days_to_expiration_date
    , greatest(.01, least(.99, (lead(price) over (w) - price) / (lead(strike) over (w) - strike))) as first_order_difference
    , file_date as file_datetime
    , ingest_datetime
  from ranked
  where rn = 1
  window w as (partition by symbol, file_date, days_to_expiration, is_call order by strike)
)
select *
from final
