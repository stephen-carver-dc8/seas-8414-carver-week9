# Dockerfile
# Use Python 3.11
FROM python:3.11-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# 1. Install system-level build dependencies

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential openjdk-17-jre-headless ca-certificates curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Copy only the requirements file to leverage Docker's layer caching.
# This layer only gets rebuilt if requirements.txt changes.
COPY requirements.txt .

# 3. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit runs on
EXPOSE 54321

# The CMD to start the application is located in the docker-compose.yml file
# for better development/production flexibility.

