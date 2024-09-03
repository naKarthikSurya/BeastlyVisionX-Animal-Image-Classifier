# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary project folders into the container
COPY final_model /app/final_model
COPY model /app/model
COPY main /app/main
COPY modules /app/modules

# Copy the model.safetensors file into the final_model directory
COPY final_model/model.safetensors /app/final_model/model.safetensors

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main.main:app", "--host", "0.0.0.0", "--port", "8000"]
