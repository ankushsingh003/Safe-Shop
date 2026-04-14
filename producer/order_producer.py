import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer
from faker import Faker

# Initialize Faker
fake = Faker()

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = 'localhost:29092' # Using the host-accessible port
TOPIC_NAME = 'orders'

def get_order():
    """Generates a random order. 10% chance of being 'suspicious' (High amount)"""
    is_suspicious = random.random() < 0.1
    
    if is_suspicious:
        amount = round(random.uniform(5000, 10000), 2) # High value order
    else:
        amount = round(random.uniform(10, 500), 2)

    return {
        "order_id": fake.uuid4(),
        "user_id": fake.user_name(),
        "product_id": f"PROD-{random.randint(100, 999)}",
        "product_name": fake.catch_phrase(),
        "category": random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Beauty']),
        "amount": amount,
        "quantity": random.randint(1, 5),
        "payment_method": random.choice(['Credit Card', 'UPI', 'PayPal', 'Debit Card']),
        "ip_address": fake.ipv4(),
        "device_type": random.choice(['mobile', 'desktop', 'tablet']),
        "location_mismatch": random.choice([0, 0, 0, 0, 1]), # 20% chance of mismatch
        "timestamp": datetime.utcnow().isoformat()
    }

def main():
    # Initialize Kafka Producer
    print(f"Connecting to Kafka at {KAFKA_BOOTSTRAP_SERVERS}...")
    producer = None
    retry_count = 0
    
    while not producer and retry_count < 5:
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
        except Exception as e:
            print(f"Connection failed ({e}). Retrying in 5 seconds...")
            time.sleep(5)
            retry_count += 1

    if not producer:
        print("Could not connect to Kafka. Exiting.")
        return

    print(f"Starting to produce messages to topic: {TOPIC_NAME}...")
    
    try:
        while True:
            order = get_order()
            producer.send(TOPIC_NAME, value=order)
            print(f"Sent Order: {order['order_id']} | Amount: ${order['amount']} | Category: {order['category']}")
            
            # Random delay between 1 to 5 seconds
            time.sleep(random.uniform(1, 5))
            
    except KeyboardInterrupt:
        print("Stopping producer...")
    finally:
        producer.close()

if __name__ == "__main__":
    main()
