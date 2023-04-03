FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update && \
    apt install --no-install-recommends -y python3-pip build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/src
WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

ADD src /app/src
ENV PYTHONPATH=/app/src

CMD ["/bin/bash", "-c", "gunicorn --workers $WORKERS --worker-class uvicorn.workers.UvicornWorker \
      --access-logfile - \
      --log-level info \
      --bind 0.0.0.0:8000 \
      --timeout 300 \
      src.main:app"]
