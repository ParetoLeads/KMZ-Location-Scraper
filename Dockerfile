FROM python:3.12-slim

WORKDIR /app

COPY requirements-railway.txt ./
RUN pip install --no-cache-dir -r requirements-railway.txt

COPY . .

CMD ["python", "start.py"]
