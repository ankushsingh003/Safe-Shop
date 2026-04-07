# 🛒 Real-Time E-Commerce Order Analytics Pipeline

> A production-grade data engineering project that ingests live e-commerce orders, detects fraudulent transactions using ML, forecasts demand, and delivers real-time business insights through a unified dashboard.

---

## 📌 Problem Statement

Large e-commerce platforms like Flipkart, Amazon, and Meesho process thousands of orders every minute. Without a real-time data infrastructure, these companies face four critical problems:

| # | Problem | Business Impact |
|---|---------|----------------|
| 1 | **Fraud Orders** — Stolen cards, fake accounts, and bot orders go undetected until damage is done | Revenue loss, chargebacks |
| 2 | **No Live Visibility** — Business teams only see yesterday's reports, not what's happening right now | Missed decisions, slow response |
| 3 | **Demand Unpredictability** — Products go out of stock during sales events like Big Billion Days | Lost revenue, poor customer experience |
| 4 | **Scattered Data** — Order data sits across 10+ systems with no unified pipeline | Inconsistent reporting, data silos |

---

## ✅ Solution

This project builds an **end-to-end real-time data pipeline** that:

- 🔴 **Ingests** live order events using Apache Kafka
- ⚡ **Processes** streams in real time using Apache Spark Structured Streaming
- 🤖 **Detects fraud** using an ML model (Isolation Forest) served via FastAPI
- 📈 **Forecasts demand** using a time-series model (Prophet / LSTM)
- 🗄️ **Stores** processed data in PostgreSQL and raw data in AWS S3 / Delta Lake
- 🔁 **Orchestrates** workflows and retraining jobs using Apache Airflow
- 📊 **Visualizes** live metrics in a Grafana dashboard

---

## 🏗️ Architecture

```
┌─────────────┐     ┌───────────┐     ┌──────────────────┐     ┌─────────────┐
│ Order        │────▶│   Kafka   │────▶│  Spark Streaming │────▶│  PostgreSQL │
│ Generator    │     │  Broker   │     │  + ML Inference  │     │  / AWS S3   │
└─────────────┘     └───────────┘     └──────────────────┘     └──────┬──────┘
                                               │                        │
                                       ┌───────▼───────┐      ┌────────▼───────┐
                                       │  FastAPI ML   │      │    Airflow     │
                                       │  Server       │      │  Orchestrator  │
                                       │  (Fraud/      │      │  (Batch Jobs / │
                                       │   Forecast)   │      │   Retraining)  │
                                       └───────────────┘      └────────┬───────┘
                                                                        │
                                                               ┌────────▼───────┐
                                                               │    Grafana     │
                                                               │   Dashboard    │
                                                               └────────────────┘
```

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Generation | Python (Faker library) |
| Message Broker | Apache Kafka |
| Stream Processing | Apache Spark Structured Streaming |
| ML Models | Scikit-learn (Isolation Forest), Prophet / LSTM |
| Model Serving | FastAPI |
| Storage (Structured) | PostgreSQL |
| Storage (Raw/Archive) | AWS S3 + Delta Lake |
| Orchestration | Apache Airflow |
| Visualization | Grafana |
| Containerization | Docker + Docker Compose |
| Model Registry | MLflow |

---

## 📂 Project Structure

```
realtime-ecommerce-pipeline/
│
├── producer/                    # Generates & sends fake orders to Kafka
├── kafka/                       # Kafka configuration & topic setup
├── spark/                       # Spark streaming & transformations
├── ml/
│   ├── train/                   # Model training scripts
│   ├── serve/                   # FastAPI inference server
│   └── models/                  # Saved model files (.pkl)
├── storage/                     # PostgreSQL & S3 writers
├── airflow/
│   └── dags/                    # Pipeline, forecast & retraining DAGs
├── dashboard/                   # Grafana configs & SQL queries
├── tests/                       # Unit & integration tests
├── docker/                      # Dockerfiles & docker-compose
├── docs/                        # Architecture diagram & setup guide
├── requirements.txt
├── .env
└── README.md
```

---

## 🤖 ML Components

### 1. Fraud Detection (Real-Time)
- **Model:** Isolation Forest
- **Trigger:** Every order event as it passes through Spark Streaming
- **Features:** `order_amount`, `orders_per_user_last_minute`, `location_mismatch`, `device_type`
- **Output:** `fraud_score` (0–1) + `is_fraud` flag written to PostgreSQL

### 2. Demand Forecasting (Batch)
- **Model:** Facebook Prophet / LSTM
- **Trigger:** Airflow DAG runs every hour
- **Features:** Historical order counts per product per region
- **Output:** Predicted order volume for next 1–6 hours

---

## 📊 Dashboard Metrics (Grafana)

- ✅ Live order count (per minute)
- ✅ Revenue by region (real-time)
- ✅ Fraud alerts feed
- ✅ Top products by order volume
- ✅ Demand forecast chart (next 6 hours)

---

## 👥 Stakeholder Impact

| Stakeholder | Benefit |
|------------|---------|
| Business Team | Live dashboard to monitor sales 24/7 |
| Finance Team | Instant fraud alerts, reduced chargebacks |
| Supply Chain | Demand forecast to manage inventory proactively |
| Data Team | Clean, unified, reliable pipeline replacing data silos |

---

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- AWS Account (for S3, optional)

### Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/realtime-ecommerce-pipeline.git
cd realtime-ecommerce-pipeline

# Set up environment variables
cp .env.example .env

# Start all services (Kafka, Spark, PostgreSQL, Airflow)
docker-compose up -d

# Start the order producer
python producer/order_producer.py

# Start the ML serving API
uvicorn ml/serve/app:app --reload --port 8000

# Access Grafana dashboard
open http://localhost:3000
```

---

## 📈 Skills Demonstrated

`Apache Kafka` `Apache Spark` `PySpark` `Apache Airflow` `FastAPI` `Scikit-learn` `Prophet` `PostgreSQL` `AWS S3` `Delta Lake` `MLflow` `Docker` `Grafana` `ETL` `Real-Time Processing` `MLOps`

---

## 🙋 Author

**Your Name**
- LinkedIn: [linkedin.com/in/yourprofile](#)
- GitHub: [github.com/yourusername](#)

---

> ⭐ If you found this project useful, consider giving it a star!