"""Stripe events asset — ingests webhooks from Kafka, lands Parquet in S3."""
from dagster import asset, AssetExecutionContext
from pipeline.io import S3ParquetIO, KafkaAvroSource


@asset(
    io_manager_key="s3_parquet",
    partitions_def="daily",
)
def stripe_events(context: AssetExecutionContext) -> None:
    """Raw Stripe events partitioned by UTC date."""
    source = KafkaAvroSource(topic="stripe.events.v2")
    for batch in source.read_batches(context.partition_key):
        S3ParquetIO.write(
            key=f"raw/stripe/{context.partition_key}/{batch.id}.parquet",
            records=batch.records,
        )
