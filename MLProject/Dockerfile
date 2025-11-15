FROM python:3.12-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir mlflow scikit-learn pandas numpy

CMD ["python", "modelling.py", "--data_path", "studentsperformance_preprocessing/StudentsPerformance_preprocessing.csv"]
