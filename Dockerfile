FROM python:3.11-slim

WORKDIR /app

# Dépendances système utiles (OpenCV, numpy, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8501

CMD ["streamlit", "run", "src/4 - streamlit/app2.py", "--server.port=8501", "--server.address=0.0.0.0"]
