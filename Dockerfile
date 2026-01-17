# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-deps -r requirements.txt

# Expose the port the app runs on
EXPOSE 7860

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "ytbot_gemini.py"]