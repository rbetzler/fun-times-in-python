--daily percent diffs
--five day diffs
select
	date(market_datetime) as market_datetime,
	symbol,
	open,
	lag(open) over (partition by symbol order by market_datetime) as yesterday_open,
	open - lag(open) over (partition by symbol order by market_datetime) as daily_diff,
	case when lag(open) over (partition by symbol order by market_datetime) <> 0
		then (open - lag(open) over (partition by symbol order by market_datetime))
			/lag(open) over (partition by symbol order by market_datetime)
		else 0 end as daily_percent_diff,
	case when lag(open, 5) over (partition by symbol order by market_datetime) <> 0
		then (open - lag(open, 5) over (partition by symbol order by market_datetime))
			/lag(open, 5) over (partition by symbol order by market_datetime)
		else 0 end as five_day_percent_diff
from td.equities_view
order by symbol, market_datetime
limit 1000