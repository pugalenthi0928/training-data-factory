FROM python:3.11-slim

WORKDIR /app

# System dependencies (optional but useful for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install package and common deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . streamlit pandas altair pyyaml tqdm rouge-score

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
