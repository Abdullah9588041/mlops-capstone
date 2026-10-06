# Monitor image: synthetic traffic generator + drift checks.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY monitoring/ ./monitoring/

# Infinite traffic loop by default; override with --n-requests in compose if wanted.
CMD ["python", "monitoring/traffic.py", "--api-url", "http://api:8000", "--n-requests", "0", "--interval", "2.0", "--seed", "11"]
