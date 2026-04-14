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

## 🧠 Fraud Intelligence System (Real-Time)

We have upgraded from a simple Isolation Forest to a State-of-the-Art **4-Layer Fraud Roadmap**:

| Layer | Component | Technology | Description |
|-------|-----------|------------|-------------|
| **L1** | **Ring Detection** | GNN (GraphSAGE) | Detects fraud rings by connecting orders sharing IP/Device. |
| **L2** | **Agentic AI** | LangGraph | Borderline cases trigger an AI Investigator to research and decide. |
| **L3** | **Feature Store** | Redis | sub-3ms lookup of user reputation and real-time velocity. |
| **L4** | **Ensemble Model** | XGBoost + LightGBM | High-precision stacked ensemble with SHAP explanations. |

### Prediction Workflow:
1. **Ingest**: Spark captures order and computes basic session velocity.
2. **Inference**: FastAPI receives the order and enriches it with L3 (Redis) features.
3. **Score**: L1 (GNN) and L4 (Ensemble) provide probability scores.
4. **Investigate**: If fraud score is borderline (0.4–0.8), L2 (Agentic AI) is triggered to perform "Reasoning-over-Evidence".
5. **Explain**: SHAP values generate human-readable reasons for every block (e.g., "High-risk proxy + New account + Velocity spike").

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