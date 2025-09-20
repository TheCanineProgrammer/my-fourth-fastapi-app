# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching layers)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port (FastAPI default: 8000)
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "Fourth_Scenario:app", "--host", "0.0.0.0", "--port", "8000"]