# 🧱 Databricks Migration Guide

Moving from local Docker Spark to Databricks involves shifting from "Container-managed" to "Platform-managed" Spark.

## 1. Environment Setup
Instead of a `Dockerfile`, Databricks uses **Clusters**. 
- **DBR Version**: Select **Databricks Runtime 14.x (Spark 3.5.x, Scala 2.12)** to match your code.
- **Libraries**:
  - Install `kafka-python-ng` via PyPI.
  - Add the Maven coordinator for Kafka: `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0`.
  - Add the Postgres driver: `org.postgresql:postgresql:42.7.2`.

## 2. Deploying the Spark Job
You don't run `spark-submit` yourself. You use **Workflows**:
1.  **Create Job**: In Databricks, go to **Workflows > Create Job**.
2.  **Task Type**: Choose **Python Script**.
3.  **Upload Code**: Upload `spark/spark_processor.py` to your Workspace or link your GitHub repo.
4.  **Parameters**: Set Environment Variables (like `KAFKA_BOOTSTRAP_SERVERS`) in the Cluster configuration.

## 3. Secret Management in Databricks
Since you're avoiding AWS Secrets Manager, use **Databricks Secrets**:
```bash
# Using Databricks CLI
databricks secrets create-scope --scope pipeline-secrets
databricks secrets put --scope pipeline-secrets --key ML_API_KEY
```
In your code, you can fetch them using:
```python
api_key = dbutils.secrets.get(scope="pipeline-secrets", key="ML_API_KEY")
```

## 4. Scaling
Databricks offers **Autoscaling**. You can set your cluster to scale from 2 to 20 workers automatically during high-load periods like a "Big Billion Day" sale.

## 5. Network Connectivity
Ensure your Databricks cluster has **VPC Peering** or a **NAT Gateway** configured to access your Kafka broker and Postgres DB if they are outside the Databricks environment.
