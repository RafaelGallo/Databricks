"""Weekly fraud model retraining DAG / DAG semanal de retreino do modelo de fraude."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator

with DAG(
    dag_id="retreino_fraude_lightgbm",
    start_date=datetime(2026, 10, 1),
    schedule="@weekly",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["fraude", "mlops"],
) as dag:
    # step 1 runs the etl job, step 2 retrains / etapa 1 roda o job de etl, etapa 2 retreina
    etl = DatabricksRunNowOperator(task_id="etl_bronze_silver_gold", databricks_conn_id="databricks_default", job_id=111)
    retreino = DatabricksRunNowOperator(task_id="retreino_lightgbm", databricks_conn_id="databricks_default", job_id=222)

    etl >> retreino