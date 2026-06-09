# SafeShop GCP Cloud Run Deployment Script
# This script builds the ML API Docker image using Cloud Build and deploys it to Google Cloud Run.

$PROJECT_ID = gcloud config get-value project

if (-not $PROJECT_ID) {
    Write-Host "[ERROR] No GCP project is currently set." -ForegroundColor Red
    Write-Host "[INFO] Please run: gcloud config set project your-project-id" -ForegroundColor Yellow
    exit 1
}

Write-Host "[INFO] Starting deployment to GCP Project: $PROJECT_ID" -ForegroundColor Cyan

# 1. Parse local .env file if it exists
$envVars = @()
if (Test-Path .env) {
    Write-Host "[INFO] Found .env file. Loading environment variables..." -ForegroundColor Cyan
    Get-Content .env | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            # Extract key and value
            $idx = $line.IndexOf("=")
            $key = $line.Substring(0, $idx).Trim()
            $val = $line.Substring($idx + 1).Trim()
            
            # Avoid passing local database URLs or redis hosts that won't resolve in Cloud Run
            if ($key -ne "DATABASE_URL" -and $key -ne "REDIS_HOST" -and $key -ne "KAFKA_BOOTSTRAP_SERVERS" -and $key -ne "ML_API_URL" -and $key -ne "ML_SERVER_URL") {
                $envVars += "$key=$val"
            }
        }
    }
}

# Add default fallbacks if not in .env
if (-not ($envVars -like "API_KEY=*")) {
    $envVars += "API_KEY=dev-secret-key"
}

$envVarString = $envVars -join ","

# 2. Build the image using GCP Cloud Build
Write-Host "[INFO] Step 1: Building Docker image using Cloud Build..." -ForegroundColor Yellow

# Temporarily copy Dockerfile.ml to root as 'Dockerfile' to ensure compatibility across all gcloud versions
Copy-Item docker/Dockerfile.ml Dockerfile

try {
    gcloud builds submit --tag "gcr.io/$PROJECT_ID/safeshop-ml-api" .
}
finally {
    if (Test-Path Dockerfile) {
        Remove-Item Dockerfile
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Cloud Build failed." -ForegroundColor Red
    exit 1
}

# 3. Deploy the built image to Google Cloud Run
Write-Host "[INFO] Step 2: Deploying to Google Cloud Run..." -ForegroundColor Yellow
gcloud run deploy safeshop-ml-api `
  --image "gcr.io/$PROJECT_ID/safeshop-ml-api" `
  --port 8000 `
  --platform managed `
  --region us-central1 `
  --allow-unauthenticated `
  --set-env-vars=$envVarString

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Cloud Run deployment failed." -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] Deployment completed successfully!" -ForegroundColor Green
Write-Host "[SUCCESS] Your API is now live at the Cloud Run URL shown above!" -ForegroundColor Green
