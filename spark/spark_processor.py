from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, expr, udf, sha2, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType, BooleanType, FloatType
import os
import requests
import json

# Define Schema based on producer's json schema
schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("payment_method", StringType(), True),
    StructField("ip_address", StringType(), True),
    StructField("device_type", StringType(), True),
    StructField("location_mismatch", IntegerType(), True),
    StructField("timestamp", StringType(), True)
])

# UDF for Fraud Detection (Calling FastAPI)
def detect_fraud(order_id, user_id, amount, ip, device, location_mismatch, orders_per_user_1m):
    url = os.environ.get("ML_SERVER_URL", "http://localhost:8000") + "/predict"
    try:
        payload = {
            "order_id": str(order_id),
            "user_id": str(user_id),
            "order_amount": float(amount),
            "ip_address": str(ip),
            "device_type": str(device),
            "location_mismatch": int(location_mismatch),
            "orders_per_user_last_minute": int(orders_per_user_1m)
        }
        headers = {"X-API-KEY": os.environ.get("ML_API_KEY", "dev-secret-key")}
        response = requests.post(url, json=payload, headers=headers, timeout=2)
        if response.status_code == 200:
            res = response.json()
            return (
                res.get("is_fraud", False), 
                res.get("fraud_score", 0.0),
                res.get("risk_level", "UNKNOWN"),
                res.get("reasoning", "")
            )
    except Exception as e:
        print(f"ML API Error: {e}")
        return False, 0.0, "ERROR", str(e)
    # Default fallback (e.g. status code not 200)
    return False, 0.0, "ERROR", "Incomplete response from ML Server"

# Register the UDF
fraud_udf_schema = StructType([
    StructField("is_fraud", BooleanType(), False),
    StructField("fraud_score", FloatType(), False),
    StructField("risk_level", StringType(), False),
    StructField("reasoning", StringType(), False)
])
fraud_check_udf = udf(detect_fraud, fraud_udf_schema)

def transform_orders(raw_df, schema):
    """Core transformation logic with Data Guardian (L7) & PII Masking"""
    # 1. Parse JSON
    parsed_df = raw_df.select(from_json(col("value"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("timestamp", col("timestamp").cast(TimestampType())) \
        .withWatermark("timestamp", "1 minute")

    # 2. DATA GUARDIAN (Layer 7: Data Quality)
    # Define validation rules
    is_valid_amount = (col("amount") > 0) & (col("amount") < 100000)
    is_valid_user   = col("user_id").isNotNull()
    is_valid_date   = col("timestamp") <= expr("current_timestamp()")
    
    # Split Stream: Valid Data vs Rejected (DLQ)
    valid_df = parsed_df.filter(is_valid_amount & is_valid_user & is_valid_date)
    rejected_df = parsed_df.filter(~(is_valid_amount & is_valid_user & is_valid_date))

    # 3. PII MASKING (Layer 6: Security & Compliance)
    protected_df = valid_df \
        .withColumn("ip_address_raw", col("ip_address")) \
        .withColumn("ip_address", sha2(col("ip_address"), 256)) \
        .withColumn("user_id_masked", concat_ws("-", col("user_id"), col("order_id")))

    # 4. Feature Engineering
    orders_per_user = protected_df \
        .groupBy(
            window(col("timestamp"), "1 minute", "30 seconds"),
            col("user_id")
        ).count() \
        .withColumnRenamed("count", "orders_per_user_1m")

    # 4. Join back to the main stream
    enriched_df = protected_df.join(
        orders_per_user,
        expr("""
            parsed_df.user_id = orders_per_user.user_id AND
            parsed_df.timestamp >= window.start AND
            parsed_df.timestamp < window.end
        """),
        "left"
    ).select(
        protected_df["*"],
        col("orders_per_user_1m")
    ).fillna(0)

    # 5. ML Inference: Detect Fraud
    predictions_df = enriched_df.withColumn(
        "fraud_res", 
        fraud_check_udf(
            col("order_id"), col("user_id"), col("amount"), 
            col("ip_address"), col("device_type"), col("location_mismatch"),
            col("orders_per_user_1m")
        )
    ).select("*", "fraud_res.is_fraud", "fraud_res.fraud_score", "fraud_res.risk_level", "fraud_res.reasoning") \
     .drop("fraud_res")
        
    return predictions_df, rejected_df

def main():
    # Configuration
    kafka_bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    postgres_db = os.environ.get("POSTGRES_DB", "safeshop_orders")
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    postgres_url = f"jdbc:postgresql://localhost:5432/{postgres_db}"
    
    # Initialize Spark with Delta Lake and Postgres support
    spark = SparkSession.builder \
        .appName("RealTimeOrderProcessing") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.2,io.delta:delta-spark_2.12:3.1.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # 1. Read from Kafka
    raw_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", "orders") \
        .option("startingOffsets", "latest") \
        .load() \
        .selectExpr("CAST(value AS STRING)")

    # Call the transformation logic
    predictions_df, rejected_df = transform_orders(raw_df, schema)

    # 5. Output sinks
    # a) Console sink (Successful Orders)
    console_query = predictions_df \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    # b) DLQ Sink (Rejected Orders)
    print("Starting Dead Letter Queue (DLQ) Sink...")
    dlq_query = rejected_df \
        .writeStream \
        .format("console") \
        .option("truncate", "false") \
        .start()

    # b) PostgreSQL sink
    def write_to_postgres(batch_df, batch_id):
        batch_df.write \
            .format("jdbc") \
            .option("url", postgres_url) \
            .option("dbtable", "processed_orders") \
            .option("user", postgres_user) \
            .option("password", postgres_password) \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()

    print("Starting PostgreSQL Sink...")
    postgres_query = predictions_df \
        .writeStream \
        .foreachBatch(write_to_postgres) \
        .option("checkpointLocation", "storage/checkpoints/spark_orders") \
        .start()

    # c) Delta Lake sink (Raw Data Archive)
    print("Starting Delta Lake Sink...")
    delta_query = predictions_df \
        .writeStream \
        .format("delta") \
        .outputMode("append") \
        .option("checkpointLocation", "storage/checkpoints/delta_orders") \
        .start("storage/delta/processed_orders")

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
