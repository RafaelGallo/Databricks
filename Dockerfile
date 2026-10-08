# base image with python / imagem base com python
FROM python:3.11-slim

# libgomp is required by lightgbm / libgomp é exigida pelo lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install dependencies first to use the build cache / instala as dependências primeiro para aproveitar o cache do build
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only the api code goes into the image, models are mounted at run time / só o código da api entra na imagem, os modelos são montados na execução
COPY app/main.py ./main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]