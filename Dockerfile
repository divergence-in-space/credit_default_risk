FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-ui.txt
COPY src ./src
COPY models ./models
COPY app.py start.sh ./
EXPOSE 7860
CMD ["bash", "start.sh"]
