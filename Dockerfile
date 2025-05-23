FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for textract and PyTorch
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
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create necessary directories with proper permissions
RUN mkdir -p input output cv_dummy models
RUN chmod -R 777 input output cv_dummy models

# Copy the application code
COPY . .

# Add debugging code
RUN echo "import os; print('Directories at startup:', os.listdir('/app'))" > /app/debug_startup.py
RUN echo "import glob; print('PKL files:', glob.glob('/app/output/**/*.pkl', recursive=True))" >> /app/debug_startup.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Command to run the application with debugging
CMD ["sh", "-c", "python /app/debug_startup.py && uvicorn main:app --host 0.0.0.0 --port 8000"]