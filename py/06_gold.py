# Databricks notebook source
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
df_silver

# COMMAND ----------

# cutoff at the 80th percentile of event time / corte no percentil 80 do tempo do evento
df_time = df_silver.withColumn("event_ts", F.col("event_time").cast("long"))
cutoff_ts = df_time.approxQuantile("event_ts", [0.8], 0.001)[0]
print("cutoff:", cutoff_ts)

# COMMAND ----------

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