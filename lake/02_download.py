# Databricks notebook source
"""Install the Kaggle client / instala o cliente do Kaggle."""
!pip install kaggle --quiet

"""Restart Python to load the new package / reinicia o Python para carregar o pacote novo."""
dbutils.library.restartPython()

# COMMAND ----------

"""Check that the package is available / confirma que o pacote está disponível."""

import importlib.util

# True means the kaggle package is installed / True significa que o pacote kaggle está instalado
print(importlib.util.find_spec("kaggle") is not None)

# COMMAND ----------

"""Download and extract the Kaggle dataset into bronze / baixa e extrai o dataset do Kaggle no bronze."""

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from getpass import getpass

# credentials typed at runtime, never saved in the notebook / credenciais digitadas na hora, nunca salvas no notebook
os.environ["KAGGLE_USERNAME"] = input("Kaggle username: ")
os.environ["KAGGLE_KEY"] = getpass("Kaggle key: ")

# dataset slug and volume path / identificador do dataset e caminho do volume
dataset = "karthikgangula/credit-card-fraud-mega-dataset"
volume_path = r"/Volumes/workspace/bronze/raw_files"
arquivos = ["customers.csv", "credit_card_fraud.csv"]

# authenticate / autentica
api = KaggleApi()
api.authenticate()

# download and extract only the files that are missing / baixa e extrai somente os arquivos que faltam
existentes = [f.name for f in dbutils.fs.ls(volume_path)]
for nome in arquivos:
    if nome in existentes:
        print("already in volume, skipping:", nome)
        continue
    api.dataset_download_file(dataset, nome, path=volume_path, force=True)
    zip_path = rf"{volume_path}/{nome}.zip"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(volume_path)
        print("extracted:", nome)

# COMMAND ----------

"""Confirm the files in the volume / confirma os arquivos no volume."""

# files and sizes in GB / arquivos e tamanhos em GB
for f in dbutils.fs.ls(volume_path):
    print(f.name, round(f.size / 1024**3, 3), "GB")

# COMMAND ----------

"""Confirm the files in the volume / confirma os arquivos no volume."""

# files and sizes in GB / arquivos e tamanhos em GB
for f in dbutils.fs.ls(volume_path):
    print(f.name, round(f.size / 1024**3, 3), "GB")

# COMMAND ----------

"""Peek at the first rows of the large file / espia as primeiras linhas do arquivo grande."""

import pandas as pd

# volume path / caminho do volume
volume_path = r"/Volumes/workspace/bronze/raw_files"

# read only 1000 rows, no full scan / lê só 1000 linhas, sem varredura completa
df_peek = pd.read_csv(rf"{volume_path}/credit_card_fraud.csv", nrows=1000)

# columns, types and first rows / colunas, tipos e primeiras linhas
print(df_peek.dtypes)
display(df_peek.head(10))

# COMMAND ----------

"""Inspect the customers file / inspeciona o arquivo de clientes."""

# read the small file with schema inference / lê o arquivo pequeno inferindo o schema
df_customers = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(rf"{volume_path}/customers.csv")
)
df_customers.printSchema()
display(df_customers.limit(10))
print(df_customers.count())

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

