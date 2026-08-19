FROM python:3.12-slim

WORKDIR /app

# Runtime deps only (streamlit/plotly/pandas/numpy) — all ship manylinux wheels,
# so no compiler is needed. The forecasting stack that used to require
# build-essential now lives in requirements-precompute.txt and runs offline.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Gzip Streamlit's frontend bundle at build time. Streamlit skips gzip for
# /static/ (tuned for localhost); over the internet that costs this app ~6.5 MB
# of extra transfer on every cold browser load. Doing it here keeps the win at
# zero request-time CPU.
COPY serve.py .
RUN python serve.py --precompress

# Application source + the precomputed artifacts the app actually reads.
COPY app.py .
COPY .streamlit/ ./.streamlit/
COPY 03_outputs/precomputed/ ./03_outputs/precomputed/
COPY 03_outputs/Multivariate_time_series_predictions ./03_outputs/
COPY 03_outputs/arima_vs_lstm_comparison.csv ./03_outputs/

# Cloud Run injects $PORT; Streamlit must bind to it on 0.0.0.0
ENV PORT=8080 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# serve.py installs the pre-compressed-static middleware, then hands off to
# Streamlit's own CLI, so every --server.* flag still works.
ENTRYPOINT ["sh", "-c", \
  "python serve.py \
   --server.port=$PORT \
   --server.address=0.0.0.0"]
