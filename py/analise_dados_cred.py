# Databricks notebook source
!pip install seaborn

# COMMAND ----------

"""Imports and plot style / imports e estilo dos gráficos."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pyspark.sql import functions as F

# show all columns without truncation / mostra todas as colunas sem cortar
pd.set_option("display.max_columns", None)

# color palette / paleta de cores
COR_PRIMARIA = "#0072B2"
COR_SECUNDARIA = "#94A3B8"
COR_DESTAQUE = "#E69F00"

# global plot parameters / parâmetros globais dos gráficos
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 110,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# COMMAND ----------

"""Load the silver table / carrega a tabela silver."""

# silver transactions without personal data / transações silver sem dados pessoais
df = spark.table("workspace.silver.transactions")
print(df.count(), df.columns)
display(df.limit(5))

# COMMAND ----------

"""Fraud rate by category / taxa de fraude por categoria."""

# aggregate in spark, bring only the summary to pandas / agrega no spark e leva só o resumo para o pandas
por_categoria = (
    df.groupBy("category")
    .agg(F.count("*").alias("n"), F.sum("is_fraud").alias("fraudes"), F.avg("is_fraud").alias("taxa"))
    .toPandas()
    .sort_values("taxa", ascending=False)
)
por_categoria

# COMMAND ----------

"""Target distribution / distribuição da variável alvo."""

# count each class in spark and bring the small result to pandas / conta cada classe no spark e leva o resultado pequeno para o pandas
alvo = df.groupBy("is_fraud").count().toPandas().sort_values("is_fraud")
alvo["classe"] = alvo["is_fraud"].map({0: "legítima", 1: "fraude"})
alvo["pct"] = alvo["count"] / alvo["count"].sum() * 100
alvo

# COMMAND ----------

"""Plot target distribution / gráfico da distribuição do alvo."""

# log scale keeps the small fraud class visible / escala log mantém a classe pequena de fraude visível
fig, ax = plt.subplots()
barras = ax.bar(alvo["classe"], alvo["count"], color=[COR_SECUNDARIA, COR_DESTAQUE])
ax.set_yscale("log")

# label each bar with count and percentage / rotula cada barra com contagem e percentual
for barra, n, pct in zip(barras, alvo["count"], alvo["pct"]):
    ax.text(barra.get_x() + barra.get_width() / 2, n, f"{n:,.0f}\n({pct:.2f}%)", ha="center", va="bottom")

ax.set_title("Distribuição do alvo: fraude x legítima")
ax.set_ylabel("transações (escala log)")
plt.show()

"""Class imbalance ratio / razão de desbalanceamento das classes."""

# legitimate transactions per fraud / transações legítimas para cada fraude
razao = alvo.loc[alvo["is_fraud"] == 0, "count"].iloc[0] / alvo.loc[alvo["is_fraud"] == 1, "count"].iloc[0]
print(f"1 fraude para cada {razao:,.0f} transações legítimas")

# COMMAND ----------

"""Correlation matrix on the full silver table / matriz de correlação na silver inteira."""

from itertools import combinations

# numeric columns plus the target / colunas numéricas mais o alvo
cols = ["amt", "age", "distance_km", "city_pop", "hour", "day_of_week", "month", "is_fraud"]

# all pairwise correlations in a single spark pass / todas as correlações par a par em uma única passada do spark
pares = list(combinations(cols, 2))
linha = df.select([F.corr(a, b).alias(f"{a}|{b}") for a, b in pares]).first()

# symmetric matrix with 1.0 on the diagonal / matriz simétrica com 1,0 na diagonal
corr = pd.DataFrame(1.0, index=cols, columns=cols)
for (a, b), valor in zip(pares, linha):
    corr.loc[a, b] = valor
    corr.loc[b, a] = valor
corr.round(3)

# COMMAND ----------

"""Plot correlation heatmap / gráfico do mapa de calor de correlação."""

# diverging colormap centered on zero / mapa de cores divergente centrado em zero
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
ax.set_yticks(range(len(cols)), cols)
ax.grid(False)

# write the value inside each cell / escreve o valor dentro de cada célula
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)

plt.colorbar(im, label="correlação de Pearson")
ax.set_title("Correlação entre variáveis numéricas")
plt.show()

# COMMAND ----------

"""Correlation with the target / correlação com o alvo."""

# drop the target itself and sort by absolute value / remove o próprio alvo e ordena pelo valor absoluto
com_alvo = corr["is_fraud"].drop("is_fraud").sort_values(key=abs)

plt.barh(com_alvo.index, com_alvo.values, color=COR_PRIMARIA)
plt.title("Correlação de cada variável com a fraude")
plt.show()

# COMMAND ----------

"""Plot fraud rate by category / gráfico da taxa de fraude por categoria."""

# overall fraud rate as reference line / taxa geral de fraude como linha de referência
taxa_geral = por_categoria["fraudes"].sum() / por_categoria["n"].sum() * 100

# bars sorted by rate, top category highlighted / barras ordenadas pela taxa, categoria líder destacada
dados = por_categoria.sort_values("taxa")
cores = [COR_SECUNDARIA] * (len(dados) - 1) + [COR_DESTAQUE]
plt.barh(dados["category"], dados["taxa"] * 100, color=cores)
plt.axvline(taxa_geral, color=COR_PRIMARIA, linestyle="--", label=f"média geral ({taxa_geral:.2f}%)")
plt.title("Taxa de fraude por categoria (%)")
plt.legend()
plt.show()

# COMMAND ----------

"""Stratified sample for distributions / amostra estratificada para distribuições."""

# keep all frauds and 0.6% of legitimate rows / mantém todas as fraudes e 0,6% das legítimas
amostra = df.sampleBy("is_fraud", fractions={0: 0.006, 1: 1.0}, seed=42).toPandas()
amostra["classe"] = amostra["is_fraud"].map({0: "legítima", 1: "fraude"})
amostra.groupby("classe").size().to_frame("n")

# COMMAND ----------

# log-scale bins, density normalizes each class / bins em escala log, densidade normaliza cada classe
piso = max(amostra["amt"].min(), 0.01)
bins = np.logspace(np.log10(piso), np.log10(amostra["amt"].max()), 50)
for classe, cor in [("legítima", COR_SECUNDARIA), ("fraude", COR_DESTAQUE)]:
    plt.hist(amostra.loc[amostra["classe"] == classe, "amt"], bins=bins, density=True, alpha=0.6, color=cor, label=classe)
plt.xscale("log")
plt.title("Valor da compra: fraude x legítima")
plt.legend()
plt.show()
amostra.groupby("classe")["amt"].describe()

# COMMAND ----------

"""Fraud rate by amount band / taxa de fraude por faixa de valor."""

# amount bands, numbered so they sort correctly / faixas de valor numeradas para ordenar corretamente
faixa = (
    F.when(F.col("amt") < 10, "1. até 10")
    .when(F.col("amt") < 50, "2. 10 a 50")
    .when(F.col("amt") < 100, "3. 50 a 100")
    .when(F.col("amt") < 250, "4. 100 a 250")
    .when(F.col("amt") < 500, "5. 250 a 500")
    .when(F.col("amt") < 1000, "6. 500 a 1000")
    .otherwise("7. acima de 1000")
)
por_faixa = (
    df.withColumn("faixa", faixa).groupBy("faixa")
    .agg(F.count("*").alias("n"), F.avg("is_fraud").alias("taxa"))
    .toPandas().sort_values("faixa")
)
display(por_faixa)

# COMMAND ----------

"""Plot fraud rate by amount band / gráfico da taxa de fraude por faixa de valor."""

plt.bar(por_faixa["faixa"], por_faixa["taxa"] * 100, color=COR_PRIMARIA)
plt.xticks(rotation=30)
plt.title("Taxa de fraude por faixa de valor (%)")
plt.show()

# COMMAND ----------

"""Fraud rate by hour of day / taxa de fraude por hora do dia."""

por_hora = (
    df.groupBy("hour").agg(F.count("*").alias("n"), F.avg("is_fraud").alias("taxa"))
    .toPandas().sort_values("hour")
)
display(por_hora)
plt.plot(por_hora["hour"], por_hora["taxa"] * 100, marker="o", color=COR_PRIMARIA)
plt.xticks(range(24))
plt.title("Taxa de fraude por hora do dia (%)")
plt.show()

# COMMAND ----------

"""Fraud rate by weekday and hour / taxa de fraude por dia da semana e hora."""

por_dia_hora = df.groupBy("day_of_week", "hour").agg(F.avg("is_fraud").alias("taxa")).toPandas()
matriz = por_dia_hora.pivot(index="day_of_week", columns="hour", values="taxa") * 100

# spark dayofweek starts on sunday / o dayofweek do spark começa no domingo
plt.imshow(matriz, aspect="auto", cmap="Blues")
plt.colorbar(label="% de fraude")
plt.yticks(range(7), ["dom", "seg", "ter", "qua", "qui", "sex", "sáb"])
plt.xticks(range(24))
plt.title("Taxa de fraude por dia da semana e hora (%)")
plt.show()

# COMMAND ----------

"""Distance distribution by class / distribuição da distância por classe."""

for classe, cor in [("legítima", COR_SECUNDARIA), ("fraude", COR_DESTAQUE)]:
    plt.hist(amostra.loc[amostra["classe"] == classe, "distance_km"], bins=50, density=True, alpha=0.6, color=cor, label=classe)
plt.title("Distância entre cliente e estabelecimento (km)")
plt.legend()
plt.show()
amostra.groupby("classe")["distance_km"].describe()

# COMMAND ----------

"""Fraud rate by age group / taxa de fraude por faixa de idade."""

# ten-year age groups / faixas de idade de dez em dez anos
por_idade = (
    df.withColumn("faixa_idade", F.floor(F.col("age") / 10) * 10).groupBy("faixa_idade")
    .agg(F.count("*").alias("n"), F.avg("is_fraud").alias("taxa"))
    .toPandas().sort_values("faixa_idade")
)
display(por_idade)
plt.bar(por_idade["faixa_idade"].astype(int).astype(str) + "+", por_idade["taxa"] * 100, color=COR_PRIMARIA)
plt.title("Taxa de fraude por faixa de idade (%)")
plt.show()

# COMMAND ----------

"""Fraud rate by month / taxa de fraude por mês."""

por_mes = (
    df.groupBy(F.date_trunc("month", "event_time").alias("mes"))
    .agg(F.count("*").alias("n"), F.sum("is_fraud").alias("fraudes"), F.avg("is_fraud").alias("taxa"))
    .toPandas().sort_values("mes")
)
display(por_mes)

# COMMAND ----------

plt.plot(por_mes["mes"], por_mes["taxa"] * 100, marker="o", color=COR_PRIMARIA)
plt.title("Taxa de fraude por mês (%)")
plt.show()

# COMMAND ----------

"""Fraud rate by state / taxa de fraude por estado."""

# overall rate as reference / taxa geral como referência
taxa_geral = df.agg(F.avg("is_fraud")).first()[0] * 100

por_estado = (
    df.groupBy("state").agg(F.count("*").alias("n"), F.avg("is_fraud").alias("taxa"))
    .toPandas().sort_values("taxa", ascending=False)
)
display(por_estado.head(15))

# COMMAND ----------

dados = por_estado.head(15).sort_values("taxa")
plt.barh(dados["state"], dados["taxa"] * 100, color=COR_SECUNDARIA)
plt.axvline(taxa_geral, color=COR_PRIMARIA, linestyle="--", label=f"média geral ({taxa_geral:.2f}%)")
plt.title("15 estados com maior taxa de fraude (%)")
plt.legend()
plt.show()