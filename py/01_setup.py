# Databricks notebook source
"""Health check of the compute / teste rápido de saúde do compute."""

# trivial query to confirm the session responds / consulta trivial para confirmar que a sessão responde
display(spark.range(5))

# COMMAND ----------

"""Create the raw volume in bronze / cria o volume bruto no bronze."""

# create the volume if it does not exist / cria o volume se não existir
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.bronze.raw_files")

# COMMAND ----------

"""Install the Kaggle client / instala o cliente do Kaggle."""

!pip install kaggle --quiet
dbutils.library.restartPython()

# COMMAND ----------

"""List dataset files without downloading / lista os arquivos do dataset sem baixar."""

import os
from getpass import getpass
from kaggle.api.kaggle_api_extended import KaggleApi

# credentials typed at runtime, never saved in the notebook / credenciais digitadas na hora, nunca salvas no notebook
os.environ["KAGGLE_USERNAME"] = input("Kaggle username: ")
os.environ["KAGGLE_KEY"] = getpass("Kaggle key: ")

# COMMAND ----------

# authenticate / autentica
api = KaggleApi()
api.authenticate()

# dataset slug / identificador do dataset
dataset = "karthikgangula/credit-card-fraud-mega-dataset"

# files and sizes in GB / arquivos e tamanhos em GB
for f in api.dataset_list_files(dataset).files:
    print(f.name, round(f.total_bytes / 1024**3, 2), "GB")

# COMMAND ----------

"""List the exact file names from Kaggle / lista os nomes exatos dos arquivos do Kaggle."""

# dataset slug / identificador do dataset
dataset = "karthikgangula/credit-card-fraud-mega-dataset"

# files with exact names, using repr to reveal spaces / arquivos com nomes exatos, usando repr para mostrar espaços
files = api.dataset_list_files(dataset).files
for f in files:
    print(repr(f.name), round(f.total_bytes / 1024**3, 2), "GB")

# COMMAND ----------

"""Download the small customers file / baixa o arquivo pequeno de clientes."""

# volume path / caminho do volume
volume_path = r"/Volumes/workspace/bronze/raw_files"

# exact file name from the kaggle listing / nome exato do arquivo conforme a listagem do kaggle
api.dataset_download_file(dataset, "customers.csv", path=volume_path, force=True)

# list what is in the volume / lista o que está no volume
for f in dbutils.fs.ls(volume_path):
    print(f.name, round(f.size / 1024**3, 3), "GB")

# COMMAND ----------

"""Inspect the customers file / inspeciona o arquivo de clientes."""

import zipfile

# unzip if the download came compressed / descompacta se o download veio compactado
for f in dbutils.fs.ls(volume_path):
    if f.name.endswith(".zip"):
        with zipfile.ZipFile(rf"{volume_path}/{f.name}") as z:
            z.extractall(volume_path)
        print("extracted:", f.name)

# read and show schema and rows / lê e mostra schema e linhas
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

"""Download the large transactions file / baixa o arquivo grande de transações."""

# exact file name from the kaggle listing / nome exato do arquivo conforme a listagem do kaggle
api.dataset_download_file(dataset, "credit_card_fraud.csv", path=volume_path, force=True)

# unzip if the download came compressed / descompacta se o download veio compactado
for f in dbutils.fs.ls(volume_path):
    if f.name.endswith(".zip"):
        with zipfile.ZipFile(rf"{volume_path}/{f.name}") as z:
            z.extractall(volume_path)
        print("extracted:", f.name)

# list what is in the volume / lista o que está no volume
for f in dbutils.fs.ls(volume_path):
    print(f.name, round(f.size / 1024**3, 2), "GB")

# COMMAND ----------

"""Peek at the large file without a full scan / espia o arquivo grande sem varrer tudo."""

# read only the first rows to get the columns / lê só as primeiras linhas para ver as colunas
df_peek = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(rf"{volume_path}/credit_card_fraud.csv")
    .limit(1000)
)
df_peek.printSchema()
display(df_peek.limit(10))

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

"""Remove the zip files after extraction / remove os zips depois da extração."""

# delete only the compressed copies / apaga somente as cópias compactadas
for f in dbutils.fs.ls(volume_path):
    if f.name.endswith(".zip"):
        dbutils.fs.rm(rf"{volume_path}/{f.name}")
        print("removed:", f.name)