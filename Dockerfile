# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 8080

# Define the command to run your app using Gunicorn
# This is the command that will run when the container starts
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "run:app"]