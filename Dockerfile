FROM python:3.10-slim

WORKDIR /app

# Instalamos dependencias del sistema necesarias para algunas librerías de ML
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

CMD ["python", "src/inference.py"]