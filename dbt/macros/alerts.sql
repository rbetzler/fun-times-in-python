{% macro missing_data_alert(
  table,
  key,
  datetime,
  n_expected_records
) %}

with
base as (
  select count(distinct {{ key }}) as n
  from {{ ref(table) }} as t
  where {{ datetime }} >= (select latest_business_day from {{ ref('business_days') }})
  union
  select 1
)
select '{{ table }} is missing data' as issue
from base
group by 1
having max(n) < {{ n_expected_records }}

{% endmacro %}
