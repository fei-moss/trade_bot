FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/app/profiles", "/app/data_cache"]

ENTRYPOINT ["python", "main.py"]
CMD ["--symbols", "BTC/USDT", "--bots", "10", "--generate", "25", "--regime", "v1"]
