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

# Install Python dependencies directly (no PyTorch needed)
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn==0.23.2 \
    python-multipart==0.0.6 \
    pandas==2.1.1 \
    numpy==1.26.0 \
    scikit-learn==1.3.1 \
    matplotlib==3.8.0 \
    docx2txt==0.8 \
    PyPDF2==3.0.1 \
    textract==1.6.5 \
    pydantic==2.4.2 \
    scipy==1.11.3

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
