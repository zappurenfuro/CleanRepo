FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
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

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    docx2txt \
    PyPDF2 \
    textract \
    pydantic

# Create necessary directories
RUN mkdir -p input output cv_dummy
RUN chmod -R 777 input output cv_dummy

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Run the simple API
CMD ["uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "8000"]