# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app


# Copy requirements.txt and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --timeout 5000


# Copy the entire project folder contents into the container
COPY . .

# Expose port if needed, e.g., for Streamlit (if you run a streamlit app)
EXPOSE 8504

# Define default command to run your script
CMD ["streamlit", "run", "bola.py"]
