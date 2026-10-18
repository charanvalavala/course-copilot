FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY static ./static
COPY data ./data
ENV PYTHONPATH=/app/src
ENV COURSE_DATA_DIR=/app/data/runtime
ENV COURSE_READ_ONLY=true
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn course_copilot.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
