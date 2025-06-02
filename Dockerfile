FROM python:3.9-slim


# Set environment variables to prevent Python from writing .pyc files and to buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Create a working directory
WORKDIR /ari5123-ai-trader-app


# Copy the requirements file
COPY ./requirements.txt ./


# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /ari5123-ai-trader-app/requirements.txt


# Copy the application files into the container
COPY . /ari5123-ai-trader-app


# Expose Streamlit's default port
EXPOSE 8506:8506


CMD ["streamlit", "run", "./src/main.py", "--server.port=8506"]
