# Airflow — Workflow Orchestration for ML Pipelines

**Version:** 2.10 / `apache-airflow>=2.10.0,<3.0.0`

## Purpose

Apache Airflow schedules, orchestrates, and monitors complex data pipelines as directed acyclic graphs (DAGs). In the WFB project it manages data ingestion, preprocessing jobs, training triggers, and model evaluation runs — with retries, alerting, and dependency management built in.

## Installation

```bash
pip install "apache-airflow[amazon,google,postgres]==2.10.0"
# Pin constraints
pip install "apache-airflow==2.10.0" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.0/constraints-3.11.txt"
```

Initialize the metadata database and create an admin user:

```bash
airflow db migrate
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

## Basic Usage Example

```python
# dags/wfb_training_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

default_args = {"owner": "wfb", "retries": 2, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="wfb_train",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
) as dag:

    wait_for_data = FileSensor(
        task_id="wait_for_csv",
        filepath="/data/wfb/input/latest.csv",
        poke_interval=60,
        timeout=600,
    )

    def preprocess():
        import pandas as pd
        df = pd.read_csv("/data/wfb/input/latest.csv")
        df.to_parquet("/data/wfb/processed/clean.parquet")

    preprocess_task = PythonOperator(task_id="preprocess", python_callable=preprocess)

    train_task = BashOperator(
        task_id="train_model",
        bash_command="python /wfb_model/train.py --data /data/wfb/processed/clean.parquet",
    )

    wait_for_data >> preprocess_task >> train_task
```

## Advanced Usage / Configuration

- **Sensors**: `FileSensor`, `S3KeySensor`, `ExternalTaskSensor` wait for upstream dependencies.
- **Trigger Rules**: Set `trigger_rule="all_done"` or `"one_failed"` for fan-in/fan-out patterns.
- **XCom**: Pass small values between tasks via `ti.xcom_push(key="metric", value=acc)`.
- **Pools and slots**: Reserve GPU capacity using a custom pool with `pool="gpu_pool"` and `pool_slots=1`.
- **SLAs and alerts**: Define `sla=timedelta(hours=2)` on tasks; configure `smtp` or `slack` webhook in `airflow.cfg`.
- **KubernetesPodOperator**: Run training in isolated K8s pods with GPU limits.
- **Branching**: Use `BranchPythonOperator` to conditionally skip tasks based on data quality checks.

## Integration with the WFB Model Project

Place DAGs in `wfb_model/dags/`. The project mounts shared volumes (`/data/wfb/input`, `/data/wfb/models`) for artifact passing. Training triggers from Airflow write run IDs to a PostgreSQL metadata DB, linking WFB experiment tracking with Airflow run history.

```python
# Example: Trigger training only after feature store refresh
with DAG(...) as dag:
    refresh_features = BashOperator(...)
    train = BashOperator(task_id="train", bash_command="python /wfb_model/train.py")
    refresh_features >> train
```

## Common Pitfalls / Troubleshooting

- **DAG not appearing in UI**: Verify the DAG file is in the `dags_folder` path set in `airflow.cfg`. Check `airflow dags list` and look for import errors in the Airflow UI.
- **Sensor never triggers**: Increase `poke_interval` or decrease `mode="reschedule"` to free worker slots while waiting.
- **XCom exceeds size limit**: XCom is stored in the metadata DB with a 2 KB limit for backends. Use an S3/XCom backend for large objects.
- **Scheduler lag**: Scale with `airflow scheduler --num_runs=...` or use the CeleryExecutor with a proper message broker.
- **Time zone mismatch**: Always use UTC in `start_date` and let Airflow convert; avoid naive `datetime` objects.
- **DAG timeout**: Set `dagrun_timeout=timedelta(hours=6)` to prevent orphaned runs.

## Documentation Links

- Airflow docs: https://airflow.apache.org/docs/apache-airflow/stable/
- Operators reference: https://airflow.apache.org/docs/apache-airflow/stable/_api/airflow/operators/index.html
- Best practices: https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
- KubernetesPodOperator: https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/
