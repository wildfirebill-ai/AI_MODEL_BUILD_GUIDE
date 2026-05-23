# dbt — Data Build Tool

SQL + Jinja transformation toolkit for analytics engineering. Models, tests, documentation, and lineage.

## Key Concepts

| Component | Description |
|-----------|-------------|
| **Models** | SQL SELECT statements with Jinja |
| **Sources** | Raw data table references |
| **Tests** | `unique`, `not_null`, `relationships`, custom SQL |
| **Lineage** | DAG via `ref()` function |
| **Incremental models** | Process only new/changed data |
| **Snapshots** | Type-2 SCD tracking |

## Installation

```bash
pip install dbt-core
pip install dbt-postgres  # or dbt-bigquery, dbt-snowflake, dbt-duckdb
```

## Project Structure

```
my_project/
├── dbt_project.yml
├── models/
│   ├── staging/stg_orders.sql
│   ├── staging/stg_customers.sql
│   ├── marts/dim_customers.sql
│   └── marts/fct_orders.sql
├── tests/
├── snapshots/
├── macros/
└── seeds/
```

## Staging Model

```sql
-- models/staging/stg_orders.sql
WITH source AS (
    SELECT * FROM {{ source('raw', 'orders') }}
)
SELECT
    id AS order_id,
    customer_id,
    order_date,
    amount,
    status,
    created_at::timestamp AS created_at
FROM source
WHERE id IS NOT NULL
```

## Mart & Incremental

```sql
-- models/marts/fct_orders.sql
WITH orders AS (SELECT * FROM {{ ref('stg_orders') }}),
     customers AS (SELECT * FROM {{ ref('stg_customers') }})
SELECT o.order_id, c.full_name, o.order_date, o.amount,
       SUM(o.amount) OVER (PARTITION BY o.customer_id) AS lifetime_value
FROM orders o LEFT JOIN customers c ON o.customer_id = c.customer_id

{{ config(materialized='incremental', unique_key='order_id') }}
SELECT * FROM {{ ref('stg_orders') }}
{% if is_incremental() %} WHERE created_at > (SELECT MAX(created_at) FROM {{ this }}) {% endif %}
```
```yaml
# models/sources.yml
version: 2
sources:
  - name: raw
    tables:
      - name: orders
        columns: [{name: id, tests: [unique, not_null]}]
models:
  - name: dim_customers
    columns: [{name: customer_id, tests: [unique, not_null]}]
```

## Running

```bash
dbt build; dbt run; dbt test; dbt docs generate
dbt run --select fct_orders+  # with upstream deps
```

```yaml
# dbt_project.yml
name: my_project
version: 1.0.0
config-version: 2
models:
  my_project:
    staging: {+materialized: view}
    marts: {+materialized: table}
```

## ML Feature Engineering

```sql
-- models/ml/features.sql
SELECT
    customer_id,
    COUNT(order_id) AS order_count_30d,
    SUM(amount) AS total_spend_30d,
    AVG(amount) AS avg_order_value_30d,
    DATEDIFF('day', MAX(order_date), CURRENT_DATE) AS days_since_last_order
FROM {{ ref('stg_orders') }}
WHERE order_date >= DATEADD('day', -30, CURRENT_DATE)
GROUP BY 1
```

## Resources

- [dbt Docs](https://docs.getdbt.com/)
- [GitHub](https://github.com/dbt-labs/dbt-core)
