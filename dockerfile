# 1. Use official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy dependency file first for Docker layer caching
COPY requirements.txt .

# 4. Install system packages and Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy project source code and artifacts
COPY . .

# Copy trained model and feature definitions to /app/model
COPY artifacts/model /app/model

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Start application via Uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
