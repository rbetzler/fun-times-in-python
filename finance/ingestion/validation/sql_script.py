MOVING_AVERAGE = """
INSERT INTO validation.benchmarks (
    with ma as (
        select 
            symbol, 
            market_datetime, 
            open,
            avg(open) over (
                partition by symbol
                order by market_datetime
                rows between 5 preceding and current row) as ma_five,
            avg(open) over (
                partition by symbol
                order by market_datetime
                rows between 20 preceding and current row)as ma_twenty
        from td.equities
        where open between 4 and 10000
        order by 1,2),
    lags as (
        select 
            *, 
            lag(ma_five) over win_lags as lag_ma_five,
            lag(ma_twenty) over win_lags as lag_ma_twenty
        from ma
        window win_lags as (partition by symbol order by market_datetime)),
    trans as (
        select 
            *, 
            case when ma_five > ma_twenty then 'buy' when ma_five < ma_twenty then 'sell' end as transaction_type
        from lags
        where 
        (ma_five > ma_twenty and lag_ma_five <= lag_ma_twenty)
        or (ma_five < ma_twenty and lag_ma_five >= lag_ma_twenty)),
    trades as(
        select 
            *, 
            lead(market_datetime) over win_trades as sell_date,
            case when transaction_type = 'buy' then lead(open) over win_trades end as sell_price
        from trans
        window win_trades as (partition by symbol order by market_datetime))
    select 
        market_datetime,
        'moving_average_five_twenty_days' as model_type,
        sum((sell_price - open) / open) as profit_loss,
        NOW() as ingest_datetime
    from trades
    where sell_price is not null
    group by market_datetime
    order by market_datetime)
    """