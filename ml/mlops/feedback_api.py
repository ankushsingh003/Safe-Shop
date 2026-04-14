"""
MLOps Feedback Loop API (Layer 5)
Handles Analyst feedback to correct AI decisions.
"""

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import csv
import os
from datetime import datetime

app = FastAPI(title="MLOps Feedback Loop")

FEEDBACK_LOG = "mlops_feedback.csv"

class FeedbackRequest(BaseModel):
    order_id: str
    model_prediction: bool
    actual_label: bool  # Ground truth from human analyst
    analyst_id: str
    comments: str = ""

def log_feedback(f: FeedbackRequest):
    write_header = not os.path.exists(FEEDBACK_LOG)
    with open(FEEDBACK_LOG, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["timestamp", "order_id", "prediction", "actual", "analyst", "comments"])
        writer.writerow([
            datetime.now().isoformat(),
            f.order_id,
            f.model_prediction,
            f.actual_label,
            f.analyst_id,
            f.comments
        ])

@app.post("/feedback")
async def receive_feedback(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Endpoint for analysts to submit 'Ground Truth'.
    This data is later used to retrain the model.
    """
    background_tasks.add_task(log_feedback, feedback)
    return {"status": "Feedback received. Logging for retraining pipeline."}

@app.get("/stats")
async def get_stats():
    """Returns count of corrections made by analysts."""
    if not os.path.exists(FEEDBACK_LOG):
        return {"total_feedback_entries": 0}
    with open(FEEDBACK_LOG, 'r') as f:
        rows = list(csv.reader(f))
        return {"total_feedback_entries": len(rows) - 1}

if __name__ == "__main__":
    import uvicorn
    # In production, run this alongside the serving app
    uvicorn.run(app, host="0.0.0.0", port=8001)
