FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for text extraction
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    antiword \
    unrtf \
    tesseract-ocr \
    libjpeg-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install specific numpy version that's compatible with the pickle files
RUN pip uninstall -y numpy && pip install numpy==1.24.3

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create necessary directories
RUN mkdir -p input output cv_dummy
RUN chmod -R 777 input output cv_dummy

# Copy the application code
COPY main.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
