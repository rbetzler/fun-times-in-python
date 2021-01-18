
with
b as (
  select generate_series(current_date - 7, current_date, '1 day'::interval)::date as day_date
)
select max(b.day_date) as latest_business_day
from b
left join {{ ref('holidays') }} as h
  on b.day_date = h.day_date
where h.day_date is null
  and extract(dow from b.day_date) between 1 and 5
