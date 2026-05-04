# Stage 1: build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: run the Flask app
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model.py ./
COPY --from=frontend-builder /app/dist ./frontend/dist

EXPOSE 7860
CMD ["python", "app.py"]
