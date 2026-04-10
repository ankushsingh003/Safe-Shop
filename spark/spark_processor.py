from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, count, expr, udf
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
    StructField("timestamp", StringType(), True)
])

# UDF for Fraud Detection (Calling FastAPI)
def detect_fraud(amount, orders_per_user_1m):
    url = os.environ.get("ML_SERVER_URL", "http://localhost:8000") + "/predict"
    try:
        response = requests.post(
            url, 
            json={"order_amount": float(amount), "orders_per_user_1m": float(orders_per_user_1m)},
            timeout=1
        )
        if response.status_code == 200:
            res = response.json()
            return res.get("is_fraud", False), res.get("fraud_score", 0.0)
    except Exception as e:
        print(f"ML API Error: {e}")
    # Default fallback
    return False, 0.0

# Register the UDF
# Returning a struct with (is_fraud, fraud_score)
fraud_udf_schema = StructType([
    StructField("is_fraud", BooleanType(), False),
    StructField("fraud_score", FloatType(), False)
])
fraud_check_udf = udf(detect_fraud, fraud_check_udf_schema)

def main():
    # Configuration
    kafka_bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    postgres_db = os.environ.get("POSTGRES_DB", "ecommerce_orders")
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    postgres_url = f"jdbc:postgresql://localhost:5432/{postgres_db}"
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("RealTimeOrderProcessing") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # 1. Read from Kafka
    raw_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", "orders") \
        .option("startingOffsets", "latest") \
        .load()

    # 2. Parse JSON
    parsed_df = raw_df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("timestamp", col("timestamp").cast(TimestampType())) \
        .withWatermark("timestamp", "1 minute")

    # 3. Feature Engineering: Windowed Aggregations
    orders_per_user = parsed_df \
        .groupBy(
            window(col("timestamp"), "1 minute", "30 seconds"),
            col("user_id")
        ).count() \
        .withColumnRenamed("count", "orders_per_user_1m")

    # Join back to the main stream
    enriched_df = parsed_df.join(
        orders_per_user,
        expr("""
            parsed_df.user_id = orders_per_user.user_id AND
            parsed_df.timestamp >= window.start AND
            parsed_df.timestamp < window.end
        """),
        "left"
    ).select(
        parsed_df["*"],
        col("orders_per_user_1m")
    ).fillna(0)

    # 4. ML Inference: Detect Fraud
    predictions_df = enriched_df.withColumn("fraud_res", fraud_check_udf(col("amount"), col("orders_per_user_1m"))) \
        .select("*", "fraud_res.is_fraud", "fraud_res.fraud_score") \
        .drop("fraud_res")

    # 5. Output sinks
    # a) Console sink
    console_query = predictions_df \
        .writeStream \
        .outputMode("append") \
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

    print("Starting Main Pipeline...")
    postgres_query = predictions_df \
        .writeStream \
        .foreachBatch(write_to_postgres) \
        .option("checkpointLocation", "storage/checkpoints/spark_orders") \
        .start()

    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()


