# Dockerfile for Flask Application
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port the Flask app will run on
EXPOSE 5000

# Command to run the application
CMD ["python3", "app.py"]
