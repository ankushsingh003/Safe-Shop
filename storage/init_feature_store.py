"""
Real-Time Feature Store Initialization (Layer 3)
Connects to Redis to store low-latency fraud signals.

This script demonstrates how we populate the 'Online Store' 
with user reputation and velocity metrics.
"""

import redis
import json
import time

# CONFIG
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB   = 0

def init_feature_store():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        r.ping()
        print("[DONE] Redis connection successful.")
    except Exception as e:
        print(f"[ERROR] Could not connect to Redis: {e}")
        return

    # --------------------------------------------------------------------------
    # 1. POPULATE USER REPUTATION (Static features)
    # --------------------------------------------------------------------------
    print("\nPopulating user reputation scores...")
    users = {
        "user_789": {"reputation": 0.98, "is_verified": 1, "trust_tier": "GOLD"},
        "user_101": {"reputation": 0.45, "is_verified": 0, "trust_tier": "BRONZE"},
        "user_new": {"reputation": 0.70, "is_verified": 0, "trust_tier": "SILVER"}
    }
    
    for user_id, profile in users.items():
        # Store as a Hash for structured retrieval
        r.hset(f"user:{user_id}:profile", mapping=profile)
        print(f"   Stored profile for {user_id}")

    # --------------------------------------------------------------------------
    # 2. INITIALIZE VELOCITY COUNTERS (Dynamic features)
    # --------------------------------------------------------------------------
    print("\nInitializing velocity counters...")
    # Velocity often tracked as a counter with expiration
    user_id = "user_789"
    r.set(f"user:{user_id}:velocity_1h", 1)
    r.expire(f"user:{user_id}:velocity_1h", 3600)  # Expire in 1 hour
    print(f"   Set initial velocity for {user_id}")

    # --------------------------------------------------------------------------
    # 3. VERIFY RETRIEVAL (Latency Check)
    # --------------------------------------------------------------------------
    start_time = time.time()
    val = r.hgetall("user:user_789:profile")
    end_time = time.time()
    
    print("\n--- Retrieval Test ---")
    print(f"Data for user_789: {val}")
    print(f"Latency: {(end_time - start_time)*1000:.4f} ms")
    print("----------------------")

if __name__ == "__main__":
    init_feature_store()
