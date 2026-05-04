# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db && chmod 777 /app/chroma_db

# Expose the port the app runs on
EXPOSE 8000

# Define environment variables (defaults, can be overridden)
ENV PYTHONUNBUFFERED=1
ENV PERSIST_DIRECTORY=/app/chroma_db

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
