FROM python:3.10-slim-bookworm

# Prevent Python from creating .pyc cache files inside the container
# and make logs appear immediately instead of being buffered.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# All following paths are relative to /app inside the container.
WORKDIR /app

# Runtime libraries used by OpenCV / MediaPipe on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    libgomp1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache dependency installation.
COPY requirements.txt .

# Install Python dependencies.
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy our actual FastAPI application.
# This also copies app/models/hand_landmarker.task.
COPY app ./app

# Document the port used by FastAPI/Uvicorn.
EXPOSE 8000

# Start the API.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]