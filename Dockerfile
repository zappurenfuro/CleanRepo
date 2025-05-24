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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install requests for downloading files
RUN pip install requests

# Install specific numpy version that's compatible with the pickle files
RUN pip uninstall -y numpy && pip install numpy==1.24.3

# Create necessary directories
RUN mkdir -p input output cv_dummy
RUN mkdir -p output/tfidf_enhanced_1748065889
RUN chmod -R 777 input output cv_dummy

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Run the scripts and then start the application
CMD ["sh", "-c", "python download_pickle_files.py && python copy_pickle_files.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
