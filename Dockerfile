FROM python:3.10-slim

WORKDIR /repo

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-torch.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-torch.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 5000

WORKDIR /repo/app

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", \
     "--workers", "2", "--timeout", "120", "--preload"]