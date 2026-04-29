import unittest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from spark.spark_processor import transform_orders, schema
import json
from datetime import datetime
from pyspark.sql.functions import from_json, col

class TestSparkTransformations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder \
            .appName("TestSparkTransformations") \
            .master("local[2]") \
            .getOrCreate()
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_transform_orders_logic(self):
        # Sample data
        sample_order = {
            "order_id": "test-id-1",
            "user_id": "test-user",
            "amount": 100.0,
            "timestamp": "2024-04-10T12:00:00"
        }
        
        # Create a DataFrame from the sample data
        data = [json.dumps(sample_order)]
        raw_df = self.spark.createDataFrame([(d,) for d in data], ["value"])
        
        # Run transformation (Note: Windowing/Watermarking requires streaming or specific handling)
        # For unit testing, we might need to mock the UDF or use a static version
        # Here we just verify that the parsing works as a first step
        
        parsed_df = raw_df.select(from_json(col("value"), schema).alias("data")).select("data.*")
        
        first_row = parsed_df.collect()[0]
        self.assertEqual(first_row.order_id, "test-id-1")
        self.assertEqual(first_row.amount, 100.0)

if __name__ == "__main__":
    unittest.main()
