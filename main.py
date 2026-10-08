"""FastAPI service for credit card fraud scoring / serviço FastAPI para pontuação de fraude em cartão de crédito."""

import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# model path and decision threshold come from environment variables / caminho do modelo e limiar de decisão vêm de variáveis de ambiente
MODELO_PATH = os.getenv("MODELO_PATH", "/models/smotenc/lightgbm.joblib")
LIMIAR = float(os.getenv("LIMIAR", "0.5"))

# load the full pipeline once at startup / carrega o pipeline completo uma vez na inicialização
modelo = joblib.load(MODELO_PATH)
app = FastAPI(title="cred_pred")


class Transacao(BaseModel):
    """Raw transaction features / features brutas da transação."""

    category: str
    state: str
    gender: str
    amt: float
    city_pop: int
    age: float
    distance_km: float
    hour: int
    day_of_week: int
    month: int


@app.get("/health")
def health():
    """Liveness check / checagem de disponibilidade."""
    return {"status": "ok"}


@app.post("/predict")
def predict(transacao: Transacao):
    """Fraud probability and decision / probabilidade de fraude e decisão."""
    # one row dataframe with the original columns / dataframe de uma linha com as colunas originais
    df = pd.DataFrame([transacao.model_dump()])
    proba = float(modelo.predict_proba(df)[0, 1])
    return {"probabilidade_fraude": proba, "fraude": proba >= LIMIAR, "limiar": LIMIAR}