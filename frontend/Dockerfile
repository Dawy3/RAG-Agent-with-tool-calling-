FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD python -c "import socket; s=socket.socket(); s.settimeout(1); result=s.connect_ex(('localhost', 8501)); s.close(); exit(0 if result==0 else 1)" || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

