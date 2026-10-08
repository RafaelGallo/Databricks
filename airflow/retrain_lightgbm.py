"""Retrain LightGBM with SMOTENC and replace the champion only if better / retreina o LightGBM com SMOTENC e troca o campeão somente se melhorar."""

import json
import os
import shutil
from datetime import datetime

import joblib
import mlflow
import numpy as np
from imblearn.over_sampling import SMOTENC
from lightgbm import LGBMClassifier
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

spark = SparkSession.builder.getOrCreate()

# model folder and current champion path / pasta dos modelos e caminho do campeão atual
PASTA = r"/Workspace/Users/rafaelhenriquegallo@gmail.com/Delta lake (MLops)/models/smotenc"
CAMINHO_ATUAL = rf"{PASTA}/lightgbm.joblib"
PASTA_BACKUP = rf"{PASTA}/backup"
CAMINHO_HISTORICO = rf"{PASTA}/historico_retreino.json"

# the challenger must beat the champion by this margin to avoid swaps caused by noise / o desafiante precisa vencer o campeão por esta margem para evitar trocas causadas por ruído
MARGEM = 0.002

# same features as the original training / mesmas features do treino original
cat_cols = ["category", "state", "gender"]
num_cols = ["amt", "city_pop", "age", "distance_km", "hour", "day_of_week", "month"]
features = cat_cols + num_cols

# gold tables written by the etl / tabelas gold gravadas pelo etl
df_train = spark.table("workspace.gold.train").toPandas()
df_test = spark.table("workspace.gold.test").toPandas()
X_train, y_train = df_train[features], df_train["is_fraud"]
X_test, y_test, w_test = df_test[features], df_test["is_fraud"], df_test["sample_weight"]

# encode, then oversample the training set only / codifica e depois faz oversampling somente no treino
prep = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", StandardScaler(), num_cols),
])
X_enc = prep.fit_transform(X_train)
smote = SMOTENC(categorical_features=list(range(len(cat_cols))), sampling_strategy=0.5, k_neighbors=5, random_state=42)
X_res, y_res = smote.fit_resample(X_enc, y_train)

# challenger with the same hyperparameters as the champion / desafiante com os mesmos hiperparâmetros do campeão
lgbm = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgbm.fit(X_res, y_res)
desafiante = Pipeline([("prep", prep), ("model", lgbm)])

# weighted PR-AUC on the untouched test set / PR-AUC ponderada no teste intacto
pr_desafiante = average_precision_score(y_test, desafiante.predict_proba(X_test)[:, 1], sample_weight=w_test)

# score the current champion on the same test set / avalia o campeão atual no mesmo teste
pr_atual = None
if os.path.exists(CAMINHO_ATUAL):
    try:
        atual = joblib.load(CAMINHO_ATUAL)
        pr_atual = average_precision_score(y_test, atual.predict_proba(X_test)[:, 1], sample_weight=w_test)
    except Exception as erro:
        print("could not score current champion:", erro)

# promote only if better by the margin, or if there is no usable champion / promove somente se melhor pela margem, ou se não houver campeão utilizável
promover = pr_atual is None or pr_desafiante > pr_atual + MARGEM
agora = datetime.now().strftime("%Y%m%d_%H%M%S")

if promover:
    # keep the previous champion as a timestamped backup / guarda o campeão anterior como backup com data e hora
    if os.path.exists(CAMINHO_ATUAL):
        os.makedirs(PASTA_BACKUP, exist_ok=True)
        shutil.copy2(CAMINHO_ATUAL, rf"{PASTA_BACKUP}/lightgbm_{agora}.joblib")
    joblib.dump(desafiante, CAMINHO_ATUAL, compress=3)

# log the run in mlflow / registra a execução no mlflow
mlflow.set_experiment(r"/Users/rafaelhenriquegallo@gmail.com/Delta lake (MLops)/mlflow/cred_pred")
with mlflow.start_run(run_name=f"retreino_lightgbm_{agora}"):
    mlflow.log_params({"balanceamento": "smotenc", "treino_linhas": len(X_train), "promovido": promover})
    mlflow.log_metrics({"pr_auc_desafiante": pr_desafiante, "pr_auc_atual": pr_atual if pr_atual is not None else -1.0})

# append to the retraining history / acrescenta ao histórico de retreino
historico = []
if os.path.exists(CAMINHO_HISTORICO):
    with open(CAMINHO_HISTORICO, encoding="utf-8") as f:
        historico = json.load(f)
historico.append({"data": agora, "pr_auc_desafiante": float(pr_desafiante), "pr_auc_atual": None if pr_atual is None else float(pr_atual), "promovido": bool(promover)})
with open(CAMINHO_HISTORICO, "w", encoding="utf-8") as f:
    json.dump(historico, f, indent=2)

print(f"challenger {pr_desafiante:.4f} | champion {pr_atual} | promoted: {promover}")