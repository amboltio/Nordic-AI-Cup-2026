FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Your model and the server. The sample data under data/ is not copied: only
# local_evaluator.py needs it.
COPY dtos.py utils.py example.py api.py ./

EXPOSE 9054

CMD ["python", "api.py"]
