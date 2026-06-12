FROM python:3.11-slim

WORKDIR /app

# Install curl for the docker-compose healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so pip layer is cached independently of source changes
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy all project source, results, and model checkpoint into the image
COPY . .

# Streamlit port — override with STREAMLIT_SERVER_PORT env var at runtime
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE ${STREAMLIT_SERVER_PORT}

CMD streamlit run app.py \
    --server.port=${STREAMLIT_SERVER_PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true
