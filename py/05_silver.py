# Databricks notebook source
"""Silver transformation of transactions / transformação silver das transações."""

from pyspark.sql import functions as F

# COMMAND ----------

# read the bronze table / lê a tabela bronze
df_bronze = spark.table("workspace.bronze.transactions")
df_bronze

# COMMAND ----------

# build the event timestamp from date and time / monta o timestamp do evento a partir de data e hora
df = df_bronze.withColumn(
    "event_time",
    F.to_timestamp(F.concat_ws(" ", F.col("trans_date"), F.col("trans_time"))),
)

# hash the card number so the raw value is dropped / aplica hash no número do cartão para descartar o valor original
df = df.withColumn("customer_hash", F.sha2(F.col("cc_num").cast("string"), 256))

# customer age at the time of the transaction / idade do cliente no momento da transação
df = df.withColumn(
    "age",
    F.floor(F.months_between(F.col("event_time"), F.to_date(F.col("dob"))) / 12),
)
df

# COMMAND ----------

# haversine distance between customer and merchant in km / distância haversine entre cliente e estabelecimento em km
lat1 = F.radians(F.col("lat"))
lat2 = F.radians(F.col("merch_lat"))
dlat = lat2 - lat1
dlon = F.radians(F.col("merch_long") - F.col("long"))
a = F.sin(dlat / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin(dlon / 2) ** 2
df = df.withColumn("distance_km", F.round(6371 * 2 * F.asin(F.sqrt(a)), 2))
df

# COMMAND ----------

# calendar features / features de calendário
df = (
    df.withColumn("hour", F.hour("event_time"))
    .withColumn("day_of_week", F.dayofweek("event_time"))
    .withColumn("month", F.month("event_time"))
)

# COMMAND ----------

# keep only non-identifying columns / mantém somente colunas não identificáveis
df_silver = df.select(
    "trans_num",
    "customer_hash",
    "event_time",
    "amt",
    "category",
    "merchant",
    "state",
    "city_pop",
    "gender",
    "job",
    "age",
    "distance_km",
    "hour",
    "day_of_week",
    "month",
    "is_fraud",
)

# COMMAND ----------

"""Write the silver table / grava a tabela silver."""

# write as delta, this is the heavy step, run it alone / grava em delta, é a etapa pesada, rode sozinha
df_silver.write.mode("overwrite").saveAsTable("workspace.silver.transactions")

# COMMAND ----------

"""Validate the silver table / valida a tabela silver."""

# read the silver table / lê a tabela silver
df_silver = spark.table("workspace.silver.transactions")

# row count, class balance and time range / contagem, balanceamento e intervalo de tempo
print(df_silver.count())

# COMMAND ----------

display(df_silver.groupBy("is_fraud").count())

# COMMAND ----------

display(df_silver.agg(F.min("event_time"), F.max("event_time")))

# COMMAND ----------

"""Check nulls, duplicates and value ranges / checa nulos, duplicados e faixas de valores."""

# nulls per column / nulos por coluna
display(df_silver.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df_silver.columns]))

# COMMAND ----------

# approximate distinct transaction ids versus total rows / ids de transação distintos aproximados versus total de linhas
display(df_silver.agg(F.approx_count_distinct("trans_num").alias("ids_distintos"), F.count("*").alias("linhas")))

# COMMAND ----------

# numeric summary / resumo numérico
display(df_silver.select("amt", "age", "distance_km", "city_pop").summary())

# COMMAND ----------

"""Exact duplicate check on transaction ids / checagem exata de duplicados nos ids de transação."""

from pyspark.sql import functions as F

# read the silver table / lê a tabela silver
df_silver = spark.table("workspace.silver.transactions")

# transaction ids that appear more than once / ids de transação que aparecem mais de uma vez
n_dup = df_silver.groupBy("trans_num").count().filter(F.col("count") > 1).count()
print("duplicated ids:", n_dup)

# COMMAND ----------

"""Gold layer with time split and stratified sampling / camada gold com split temporal e amostragem estratificada."""

from pyspark.sql import functions as F

# read the silver table / lê a tabela silver
df_silver = spark.table("workspace.silver.transactions")

# cutoff at the 80th percentile of event time / corte no percentil 80 do tempo do evento
df_time = df_silver.withColumn("event_ts", F.col("event_time").cast("long"))
cutoff_ts = df_time.approxQuantile("event_ts", [0.8], 0.001)[0]
print("cutoff:", cutoff_ts)

# train is before the cutoff and test is after / treino antes do corte e teste depois
df_train_full = df_time.filter(F.col("event_ts") < cutoff_ts)
df_test_full = df_time.filter(F.col("event_ts") >= cutoff_ts)

# COMMAND ----------

"""Downsample legitimate transactions and add weights / reduz as transações legítimas e adiciona pesos."""

# fraction of legitimate rows kept in each split / fração de transações legítimas mantida em cada conjunto
frac_train = 0.02
frac_test = 0.05

# keep all frauds and a fraction of legitimate rows / mantém todas as fraudes e uma fração das legítimas
df_train = df_train_full.sampleBy("is_fraud", fractions={0: frac_train, 1: 1.0}, seed=42)
df_test = df_test_full.sampleBy("is_fraud", fractions={0: frac_test, 1: 1.0}, seed=42)

# weight restores the original class proportion in metrics / o peso restaura a proporção original das classes nas métricas
df_train = df_train.withColumn("sample_weight", F.when(F.col("is_fraud") == 0, F.lit(1 / frac_train)).otherwise(F.lit(1.0)))
df_test = df_test.withColumn("sample_weight", F.when(F.col("is_fraud") == 0, F.lit(1 / frac_test)).otherwise(F.lit(1.0)))

# drop the helper column / remove a coluna auxiliar
df_train = df_train.drop("event_ts")
df_test = df_test.drop("event_ts")

# COMMAND ----------

"""Write the gold tables / grava as tabelas gold."""

# write train and test as delta / grava treino e teste em delta
df_train.write.mode("overwrite").saveAsTable("workspace.gold.train")
df_test.write.mode("overwrite").saveAsTable("workspace.gold.test")

# COMMAND ----------

"""Validate the gold tables / valida as tabelas gold."""

# size, fraud rate and time range of each split / tamanho, taxa de fraude e intervalo de tempo de cada conjunto
for nome in ["train", "test"]:
    df = spark.table(f"workspace.gold.{nome}")
    print(nome, df.count())
    display(df.groupBy("is_fraud").count())
    display(df.agg(F.min("event_time"), F.max("event_time")))