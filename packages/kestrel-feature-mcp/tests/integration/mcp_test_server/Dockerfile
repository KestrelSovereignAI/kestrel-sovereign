FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install "mcp[cli]" uvicorn starlette

# Copy server code
COPY server.py .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "server.py"]
