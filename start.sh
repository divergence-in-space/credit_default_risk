#!/bin/bash
# Runs the API (internal port 8000) and the Streamlit UI (public port 7860) together.
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
exec streamlit run app.py --server.port 7860 --server.address 0.0.0.0
