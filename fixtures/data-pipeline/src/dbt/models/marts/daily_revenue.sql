{{ config(materialized='incremental', unique_key='day') }}

with base as (
  select
    date_trunc('day', created_at) as day,
    sum(amount_cents) / 100.0 as revenue_usd,
    count(distinct customer_id) as active_customers
  from {{ ref('stg_stripe_charges') }}
  where status = 'succeeded'
  {% if is_incremental() %}
    and created_at > (select max(day) from {{ this }})
  {% endif %}
  group by 1
)

select * from base
