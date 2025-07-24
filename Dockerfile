# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Run the build script to create the chroma_db
# This will use the GOOGLE_API_KEY_1 provided during the build process
ARG GOOGLE_API_KEY_1
ENV GOOGLE_API_KEY_1=$GOOGLE_API_KEY_1
RUN python build_db.py

# Expose the port Gunicorn will run on
EXPOSE 8080

# Define the command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "run:app"]