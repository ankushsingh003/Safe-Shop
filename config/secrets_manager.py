import os
from typing import Optional

class SecretManager:
    """
    Platform-agnostic Secret Manager.
    Current: Reads from Environment Variables (Injection via .env or Docker)
    Future: Replace get_secret with a call to HashiCorp Vault or GitHub Secrets.
    """
    
    @staticmethod
    def get_secret(key: str, default: Optional[str] = None) -> str:
        """
        Retrieves a sensitive value.
        To migrate to HashiCorp Vault: 
        1. Install hvac (pip install hvac)
        2. Replace this method with vault_client.read(key)
        """
        val = os.environ.get(key, default)
        if val is None:
            # In production, we fail fast for missing critical secrets
            raise EnvironmentError(f"CRITICAL: Secret '{key}' not found in environment.")
        return val

# --------------------------------------------------------------------------
# CENTRALIZED CONFIG
# --------------------------------------------------------------------------
# All services should import these constants instead of calling os.environ
API_KEY          = SecretManager.get_secret("ML_API_KEY", "dev-secret-key")
POSTGRES_DB      = SecretManager.get_secret("POSTGRES_DB", "safeshop_orders")
POSTGRES_USER    = SecretManager.get_secret("POSTGRES_USER", "postgres")
POSTGRES_PWD     = SecretManager.get_secret("POSTGRES_PASSWORD", "postgres")
KAFKA_SERVERS    = SecretManager.get_secret("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
ML_SERVER_URL    = SecretManager.get_secret("ML_SERVER_URL", "http://localhost:8000")
