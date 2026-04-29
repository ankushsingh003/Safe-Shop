import os
import redis
import requests
import json
import logging
from kafka import KafkaProducer
import psycopg2

# Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
ML_API = "http://localhost:8000/health"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/safeshop_orders")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("system_init")

def check_redis():
    try:
        r = redis.Redis(host=REDIS_HOST, port=6379, socket_timeout=2)
        r.ping()
        logger.info("✅ Redis Feature Store (L3) is ONLINE")
        return True
    except:
        logger.error("❌ Redis is OFFLINE")
        return False

def check_kafka():
    try:
        producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP, request_timeout_ms=2000)
        logger.info("✅ Kafka Message Broker is ONLINE")
        return True
    except:
        logger.error("❌ Kafka is OFFLINE")
        return False

def check_ml_api():
    try:
        response = requests.get(ML_API, timeout=3)
        if response.status_code == 200:
            logger.info("✅ ML Intelligence API is ONLINE (Layers 1-9 Active)")
            return True
    except:
        logger.error("❌ ML Intelligence API is OFFLINE")
        return False

def check_db():
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=3)
        conn.close()
        logger.info("✅ PostgreSQL Database is ONLINE")
        return True
    except:
        logger.error("❌ PostgreSQL is OFFLINE")
        return False

def main():
    logger.info("--- STARTING PRODUCTION ENVIRONMENT READINESS CHECK ---")
    results = [check_redis(), check_kafka(), check_ml_api(), check_db()]
    
    if all(results):
        logger.info("🚀 ALL SYSTEMS NOMINAL. PIPELINE IS READY FOR MARKET DEPLOYMENT.")
    else:
        logger.warning("⚠️ SYSTEM DEGRADED. Please check the services above before running.")

if __name__ == "__main__":
    main()
