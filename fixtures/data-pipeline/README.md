# data-pipeline (fictional fixture)

An ETL + streaming pipeline that ingests events from Stripe, HubSpot and Intercom, lands them in S3 (Parquet) and Snowflake, and runs dbt transforms for the analytics warehouse.

> **Not a real project.** This fixture exists so ID-RAG parallel benchmarks can test decision recall on a plausible mid-sized data-engineering stack. All decisions in `.continuity/decisions.json` are fictional.

## Stack at a glance

- **Streaming**: Kafka (MSK) + Debezium CDC from Postgres
- **Batch orchestration**: Dagster on Kubernetes
- **Warehouse**: Snowflake + dbt
- **Lakehouse**: S3 + Parquet, cataloged by AWS Glue
- **Connectors**: Airbyte OSS for SaaS source ingestion
- **Quality**: Soda Core (replaced Great Expectations)
- **Serialization**: Protobuf (replaced Avro)
- **IaC**: Terraform

## Supersede chains (important)

- `pipeline-serialization-avro` → superseded by `pipeline-serialization-protobuf`
- `pipeline-dq-great-expectations` → superseded by `pipeline-dq-soda-core`
