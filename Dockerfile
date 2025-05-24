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

# Install dependencies but exclude sentence-transformers to avoid the huggingface_hub issue
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y sentence-transformers huggingface_hub \
    && pip install --no-cache-dir scikit-learn pandas numpy matplotlib fastapi uvicorn python-multipart docx2txt PyPDF2 textract

# Create necessary directories with proper permissions
RUN mkdir -p input output cv_dummy models
RUN chmod -R 777 input output cv_dummy models

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Command to run the lightweight API
CMD ["uvicorn", "lightweight_api:app", "--host", "0.0.0.0", "--port", "8000"]