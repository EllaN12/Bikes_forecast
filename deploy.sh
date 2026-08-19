#!/usr/bin/env bash
# Manual one-shot deploy to Cloud Run (no Cloud Build required).
# Usage: ./deploy.sh [PROJECT_ID] [REGION]
# Example: ./deploy.sh my-gcp-project us-central1

set -euo pipefail

PROJECT_ID=${1:-$(gcloud config get-value project)}
REGION=${2:-us-central1}
SERVICE=bikes-forecast
REPO=bikes-forecast
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}"

echo "▶ Project : ${PROJECT_ID}"
echo "▶ Region  : ${REGION}"
echo "▶ Image   : ${IMAGE}"
echo ""

# 0. Enable required GCP APIs (non-interactive; avoids gcloud y/N prompts)
echo "── Enabling required APIs ──"
gcloud services enable artifactregistry.googleapis.com run.googleapis.com \
  --project="${PROJECT_ID}" --quiet

# 1. Ensure Artifact Registry repo exists
echo "── Creating Artifact Registry repo (idempotent) ──"
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" --quiet 2>/dev/null || true

# 2. Authenticate Docker with Artifact Registry
echo "── Configuring Docker auth ──"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 3. Build image (linux/amd64 — required for Cloud Run when building on Apple Silicon)
echo "── Building Docker image (linux/amd64) ──"
docker build --platform linux/amd64 -t "${IMAGE}:latest" .

# 4. Push to Artifact Registry
echo "── Pushing image ──"
docker push "${IMAGE}:latest"

# 5. Deploy to Cloud Run
echo "── Deploying to Cloud Run ──"
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}:latest" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --memory=512Mi \
  --cpu=1 \
  --timeout=120 \
  --concurrency=5 \
  --cpu-boost \
  --session-affinity \
  --min-instances=0 \
  --max-instances=2 \
  --project="${PROJECT_ID}"

echo ""
echo "✅ Deployed. Service URL:"
gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)"
