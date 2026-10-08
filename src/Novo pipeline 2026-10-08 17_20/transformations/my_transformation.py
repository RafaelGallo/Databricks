"""ETL for credit card fraud, bronze to silver to gold / ETL de fraude em cartão de crédito, bronze para silver para gold."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

# spark session, works in notebooks and in job script tasks / sessão spark, funciona em notebooks e em tarefas de script de job
spark = SparkSession.builder.getOrCreate()

# paths and table names / caminhos e nomes das tabelas
VOLUME_PATH = r"/Volumes/workspace/bronze/raw_files"
TABELA_BRONZE_TX = "workspace.bronze.transactions"
TABELA_BRONZE_CLIENTES = "workspace.bronze.customers"
TABELA_SILVER = "workspace.silver.transactions"
TABELA_TREINO = "workspace.gold.train"
TABELA_TESTE = "workspace.gold.test"

# switches, the bronze csv read is the heaviest step / chaves, a leitura do csv bronze é a etapa mais pesada
RECRIAR_BRONZE = False
DEDUPLICAR = False

# fraction of legitimate rows kept in each split / fração de transações legítimas mantida em cada conjunto
FRAC_TREINO = 0.02
FRAC_TESTE = 0.05
PERCENTIL_CORTE = 0.8

if RECRIAR_BRONZE:
    # explicit schema avoids inference over 10 GB, row_id replaces Unnamed: 0 / schema explícito evita inferência em 10 GB, row_id substitui Unnamed: 0
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

    # transactions are comma separated / as transações são separadas por vírgula
    (
        spark.read.option("header", True).schema(schema_tx)
        .csv(rf"{VOLUME_PATH}/credit_card_fraud.csv")
        .write.mode("overwrite").saveAsTable(TABELA_BRONZE_TX)
    )

    # customers are pipe separated, all columns as in the transactions file / clientes são separados por pipe, colunas iguais às do arquivo de transações
    schema_clientes = StructType([
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
    (
        spark.read.option("header", True).option("sep", "|").schema(schema_clientes)
        .csv(rf"{VOLUME_PATH}/customers.csv")
        .write.mode("overwrite").saveAsTable(TABELA_BRONZE_CLIENTES)
    )

# silver starts from the bronze delta table / a silver parte da tabela delta bronze
df = spark.table(TABELA_BRONZE_TX)

# optional dedup by transaction id, it triggers a shuffle / deduplicação opcional por id de transação, dispara um shuffle
if DEDUPLICAR:
    df = df.dropDuplicates(["trans_num"])

# event timestamp from date and time / timestamp do evento a partir de data e hora
df = df.withColumn("event_time", F.to_timestamp(F.concat_ws(" ", F.col("trans_date"), F.col("trans_time"))))

# hash replaces the card number so the raw value is dropped / o hash substitui o número do cartão para descartar o valor original
df = df.withColumn("customer_hash", F.sha2(F.col("cc_num").cast("string"), 256))

# customer age at the time of the transaction / idade do cliente no momento da transação
df = df.withColumn("age", F.floor(F.months_between(F.col("event_time"), F.to_date(F.col("dob"))) / 12))

# haversine distance between customer and merchant in km / distância haversine entre cliente e estabelecimento em km
lat1 = F.radians(F.col("lat"))
lat2 = F.radians(F.col("merch_lat"))
dlat = lat2 - lat1
dlon = F.radians(F.col("merch_long") - F.col("long"))
a = F.sin(dlat / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin(dlon / 2) ** 2
df = df.withColumn("distance_km", F.round(6371 * 2 * F.asin(F.sqrt(a)), 2))

# calendar features / features de calendário
df = (
    df.withColumn("hour", F.hour("event_time"))
    .withColumn("day_of_week", F.dayofweek("event_time"))
    .withColumn("month", F.month("event_time"))
)

# keep only non identifying columns / mantém somente colunas não identificáveis
df_silver = df.select(
    "trans_num", "customer_hash", "event_time", "amt", "category", "merchant", "state",
    "city_pop", "gender", "job", "age", "distance_km", "hour", "day_of_week", "month", "is_fraud",
)
df_silver.write.mode("overwrite").saveAsTable(TABELA_SILVER)

# gold reads the saved silver table / a gold lê a tabela silver salva
df_silver = spark.table(TABELA_SILVER).withColumn("event_ts", F.col("event_time").cast("long"))

# time split first, so the future never leaks into train / split temporal primeiro, para o futuro nunca vazar para o treino
corte = df_silver.approxQuantile("event_ts", [PERCENTIL_CORTE], 0.001)[0]
df_treino = df_silver.filter(F.col("event_ts") < corte)
df_teste = df_silver.filter(F.col("event_ts") >= corte)

# keep every fraud and a fraction of legitimate rows / mantém todas as fraudes e uma fração das legítimas
df_treino = df_treino.sampleBy("is_fraud", fractions={0: FRAC_TREINO, 1: 1.0}, seed=42)
df_teste = df_teste.sampleBy("is_fraud", fractions={0: FRAC_TESTE, 1: 1.0}, seed=42)

# weight restores the original class proportion in metrics / o peso restaura a proporção original das classes nas métricas
df_treino = df_treino.withColumn("sample_weight", F.when(F.col("is_fraud") == 0, F.lit(1 / FRAC_TREINO)).otherwise(F.lit(1.0))).drop("event_ts")
df_teste = df_teste.withColumn("sample_weight", F.when(F.col("is_fraud") == 0, F.lit(1 / FRAC_TESTE)).otherwise(F.lit(1.0))).drop("event_ts")

df_treino.write.mode("overwrite").saveAsTable(TABELA_TREINO)
df_teste.write.mode("overwrite").saveAsTable(TABELA_TESTE)

# final check of size and fraud count per split / checagem final de tamanho e contagem de fraudes por conjunto
for tabela in [TABELA_SILVER, TABELA_TREINO, TABELA_TESTE]:
    df_check = spark.table(tabela)
    print(tabela, df_check.count(), df_check.filter("is_fraud = 1").count())