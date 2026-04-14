"""
MLOps Drift Detection (Layer 5)
Uses 'Evidently' to monitor data and prediction drift.

This script compares training data (reference) vs production data (current)
to detect if the fraud patterns have changed.
"""

import pandas as pd
import numpy as np
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# CONFIG
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def detect_drift():
    print("--- MLOps: Fraud Detection Drift Monitor ---")
    
    # 1. GENERATE REFERENCE DATA (Original Training Dist)
    reference_data = pd.DataFrame({
        "order_amount": np.random.lognormal(4.5, 0.8, 1000),
        "velocity": np.random.poisson(0.3, 1000),
        "is_fraud": np.random.binomial(1, 0.02, 1000)
    })

    # 2. GENERATE CURRENT DATA (Production Dist with Drift)
    # Drift: Order amounts suddenly increase (e.g. currency change or bot attack)
    # Velocity: Spikes due to a coordinated attack
    current_data = pd.DataFrame({
        "order_amount": np.random.lognormal(5.2, 0.9, 1000), 
        "velocity": np.random.poisson(1.5, 1000),
        "is_fraud": np.random.binomial(1, 0.08, 1000)
    })

    print("Analyzing drift between training and production data...")

    # 3. CREATE EVIDENTLY REPORT
    drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    drift_report.run(reference_data=reference_data, current_data=current_data)

    # 4. SAVE REPORT
    report_path = os.path.join(REPORTS_DIR, "drift_report.html")
    drift_report.save_html(report_path)
    
    print(f"\n[DONE] Drift Analysis complete.")
    print(f"Report saved to: {os.path.abspath(report_path)}")
    print("In production, this would trigger a model retraining pipeline if drift > threshold.")

if __name__ == "__main__":
    detect_drift()
