import requests
import json
import os
import logging

logger = logging.getLogger("fraud_alerts")

class AlertBot:
    """
    Service to send real-time notifications to Slack/Discord/Teams.
    Enables immediate human response to Critical Fraud events.
    """
    
    def __init__(self):
        # In a real setup, these URLs would be in HashiCorp Vault / Secrets Manager
        self.webhook_url = os.environ.get("ALERT_WEBHOOK_URL", "")
        self.enabled = bool(self.webhook_url)

    def send_fraud_alert(self, order_id: str, score: float, risk: str, reason: str):
        """Sends a rich notification to the configured channel."""
        if not self.enabled:
            logger.warning(f"Alert triggered for Order {order_id} but no Webhook URL is configured.")
            return

        # Formatting for Slack/Discord
        payload = {
            "username": "Fraud Intelligence Bot",
            "icon_emoji": ":shield:",
            "attachments": [
                {
                    "fallback": f"🚨 CRITICAL FRAUD DETECTED: {order_id}",
                    "color": "#ff0000" if risk == "CRITICAL" else "#ffa500",
                    "title": f"🚨 {risk} Risk Fraud Detected",
                    "fields": [
                        {"title": "Order ID", "value": order_id, "short": True},
                        {"title": "Fraud Score", "value": f"{score:.4f}", "short": True},
                        {"title": "Risk Level", "value": risk, "short": True},
                        {"title": "Reasoning", "value": reason, "short": False}
                    ],
                    "footer": "Fraud Intelligence Engine v5.0",
                    "ts": None # Can add timestamp here
                }
            ]
        }

        try:
            response = requests.post(
                self.webhook_url, 
                data=json.dumps(payload),
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            if response.status_code != 200:
                logger.error(f"Failed to send alert: {response.text}")
        except Exception as e:
            logger.error(f"Alert Bot Error: {e}")

# Global Instance
alert_bot = AlertBot()
