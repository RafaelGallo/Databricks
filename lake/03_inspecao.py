# Databricks notebook source
"""Peek at the first rows of the transactions file / espia as primeiras linhas do arquivo de transações."""

import pandas as pd

# COMMAND ----------

# volume path / caminho do volume
volume_path = r"/Volumes/workspace/bronze/raw_files"

# COMMAND ----------

# read only 1000 rows, no full scan / lê só 1000 linhas, sem varredura completa
df_peek = pd.read_csv(rf"{volume_path}/credit_card_fraud.csv", nrows=1000)

# columns, types and first rows / colunas, tipos e primeiras linhas
print(df_peek.dtypes)

# COMMAND ----------

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

# COMMAND ----------

print(df_customers.count())

# COMMAND ----------

