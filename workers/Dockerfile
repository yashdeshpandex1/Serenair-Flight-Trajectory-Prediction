FROM python:3.10-slim

WORKDIR /Serenair

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY ./database/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Command to keep the container running or run your script
# CMD ["python", "database/your_ingestion_script.py"]