# ---------------------------
# Base Image
# ---------------------------
FROM python:3.9-slim

# ---------------------------
# Environment Variables
# ---------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_ENV=development

# ---------------------------
# Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# Install system dependencies
# ---------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    python -m pip install --upgrade pip setuptools wheel

# ---------------------------
# Copy requirements and install
# ---------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Copy project files
# ---------------------------
COPY . .

# ---------------------------
# Expose Flask port
# ---------------------------
EXPOSE 5000

# ---------------------------
# Run Flask with hot reload
# ---------------------------
CMD ["flask", "run"]
