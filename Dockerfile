# Use slim Python base — no CUDA needed on Render (CPU inference)
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by Pillow / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .

# Install Python dependencies
# torch CPU-only wheel keeps image size ~800 MB instead of ~5 GB
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into /app
COPY App.py .
COPY index.html .
COPY best_generator.pth .


# Render injects $PORT at runtime; default to 8000 for local testing
ENV PORT=8000
ENV MODEL_PATH=/app/best_generator.pth

EXPOSE 8000

# Start the FastAPI server
CMD uvicorn App:app --host 0.0.0.0 --port ${PORT}