# Databricks notebook source
"""Bronze ingestion of customers / ingestão bronze dos clientes."""

from pyspark.sql.types import (StructType, StructField, StringType, LongType, IntegerType, DoubleType,)

# COMMAND ----------

# volume path / caminho do volume
volume_path = r"/Volumes/workspace/bronze/raw_files"

# COMMAND ----------

# explicit schema for customers / schema explícito dos clientes
schema_customers = StructType([
    StructField("ssn", StringType()),
    StructField("cc_num", LongType()),
    StructField("first", StringType()),
    StructField("last", StringType()),
    StructField("gender", StringType()),
    StructField("street", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("zip", IntegerType()),
    StructField("lat", DoubleType()),
    StructField("long", DoubleType()),
    StructField("city_pop", IntegerType()),
    StructField("job", StringType()),
    StructField("dob", StringType()),
    StructField("acct_num", LongType()),
    StructField("profile", StringType()),
])

# read the pipe-separated file / lê o arquivo separado por pipe
df_customers = (
    spark.read
    .option("header", True)
    .option("sep", "|")
    .schema(schema_customers)
    .csv(rf"{volume_path}/customers.csv")
)

# COMMAND ----------

# write the delta table / grava a tabela delta
df_customers.write.mode("overwrite").saveAsTable("workspace.bronze.customers")
print(spark.table("workspace.bronze.customers").count())

# COMMAND ----------

"""Bronze ingestion of transactions / ingestão bronze das transações."""

# explicit schema for transactions, row_id replaces Unnamed: 0 / schema explícito das transações, row_id substitui Unnamed: 0
schema_tx = StructType([
    StructField("row_id", LongType()),
    StructField("ssn", StringType()),
    StructField("cc_num", LongType()),
    StructField("first", StringType()),
    StructField("last", StringType()),
    StructField("gender", StringType()),
    StructField("street", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("zip", IntegerType()),
    StructField("lat", DoubleType()),
    StructField("long", DoubleType()),
    StructField("city_pop", IntegerType()),
    StructField("job", StringType()),
    StructField("dob", StringType()),
    StructField("acct_num", LongType()),
    StructField("profile", StringType()),
    StructField("trans_num", StringType()),
    StructField("trans_date", StringType()),
    StructField("trans_time", StringType()),
    StructField("unix_time", LongType()),
    StructField("category", StringType()),
    StructField("amt", DoubleType()),
    StructField("is_fraud", IntegerType()),
    StructField("merchant", StringType()),
    StructField("merch_lat", DoubleType()),
    StructField("merch_long", DoubleType()),
])

# COMMAND ----------

# read the comma-separated file with the explicit schema / lê o arquivo separado por vírgula com o schema explícito
df_tx = (
    spark.read
    .option("header", True)
    .schema(schema_tx)
    .csv(rf"{volume_path}/credit_card_fraud.csv")
)

# COMMAND ----------

# write the delta table / grava a tabela delta
df_tx.write.mode("overwrite").saveAsTable("workspace.bronze.transactions")

# COMMAND ----------

"""Validate the bronze table / valida a tabela bronze."""

# row count and class balance from the delta table / contagem e balanceamento de classes a partir da tabela delta
df_bronze = spark.table("workspace.bronze.transactions")
print(df_bronze.count())

# COMMAND ----------

display(df_bronze.groupBy("is_fraud").count())

# COMMAND ----------

"""Check for nulls in key columns / checa nulos nas colunas principais."""

from pyspark.sql import functions as F

# nulls in the columns that matter most / nulos nas colunas mais importantes
colunas = ["cc_num", "trans_num", "trans_date", "trans_time", "amt", "is_fraud", "merchant", "category"]
display(df_bronze.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in colunas]))