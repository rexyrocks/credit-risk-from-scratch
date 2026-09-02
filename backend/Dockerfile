# Backend Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Copy the entire project (backend needs access to the root-level
# logistic_regression.py, preprocessing.py, and evaluate.py)
COPY logistic_regression.py preprocessing.py evaluate.py ./
COPY backend/ ./backend/
COPY data/ ./data/

RUN pip install --no-cache-dir -r backend/requirements.txt

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
