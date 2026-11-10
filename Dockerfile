FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash prometheus-node-exporter \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=app:app app.py start.sh ./

USER app

EXPOSE 7860 8000 9100

ENTRYPOINT ["/bin/bash", "/app/start.sh"]
CMD ["python", "app.py"]
