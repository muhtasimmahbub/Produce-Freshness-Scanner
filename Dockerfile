FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU-only PyTorch to avoid 2.5GB CUDA bloat on Render
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-10000}"]