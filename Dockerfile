FROM python:3.12-slim

WORKDIR /app

COPY requirements-railway.txt ./
RUN pip install --no-cache-dir -r requirements-railway.txt

COPY . .

CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info"]
