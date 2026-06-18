FROM python:3.12-slim

# libgomp1 is needed by scikit-learn; build-essential for sktime/pmdarima C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying source so layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app.py .
COPY 02_SRC/ ./02_SRC/
COPY 00_data_raw/ ./00_data_raw/
COPY 01_database/ ./01_database/
COPY 04_outputs/ ./04_outputs/

# Cloud Run injects $PORT; Streamlit must bind to it on 0.0.0.0
ENV PORT=8080

EXPOSE 8080

ENTRYPOINT ["sh", "-c", \
  "streamlit run app.py \
   --server.port=$PORT \
   --server.address=0.0.0.0 \
   --server.headless=true \
   --server.enableCORS=false \
   --server.enableXsrfProtection=false"]
