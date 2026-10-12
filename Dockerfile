FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-api.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt -r requirements-ui.txt
COPY src ./src
COPY models ./models
COPY app.py start.sh ./
RUN useradd -m -u 1000 user && chown -R user /app
USER user
ENV HOME=/home/user STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 7860
CMD ["bash", "start.sh"]
