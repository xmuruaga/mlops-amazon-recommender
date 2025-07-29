from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.providers.mlflow.hooks.mlflow import MlflowHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import os

@dag(
    dag_id="recommender_pipeline",
    start_date=datetime(2025, 7, 22),
    schedule_interval="@daily",
    catchup=False,
    default_args={"owner": "airflow", "retries": 1},
)
def recommender_pipeline():

    @task(retries=3, retry_delay=timedelta(minutes=5))
    def raw_to_bronze(ds=None):
        import pandas as pd
        import os
        raw_dir = Variable.get("raw_dir")
        bronze_root = Variable.get("bronze_root")
        raw_csv_path = os.path.join(raw_dir, "Reviews.csv")
        bronze_dir = os.path.join(bronze_root, ds)
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(bronze_dir, exist_ok=True)
        bronze_path = os.path.join(bronze_dir, "reviews.parquet")
        success_flag = os.path.join(bronze_dir, "_SUCCESS")
        if os.path.exists(success_flag) and os.path.getsize(success_flag) > 0:
            return bronze_path
        # Download logic using S3Hook
        s3_conn_id = Variable.get("s3_conn_id", default_var=None)
        if s3_conn_id:
            s3 = S3Hook(aws_conn_id=s3_conn_id)
            s3_key = Variable.get("s3_reviews_key")
            s3_bucket = Variable.get("s3_bucket")
            s3.get_key(s3_key, s3_bucket).download_file(raw_csv_path)
        elif not os.path.exists(raw_csv_path):
            raise FileNotFoundError("Reviews.csv not found and no S3 connection configured.")
        try:
            df = pd.read_csv(raw_csv_path, parse_dates=["Time"], date_parser=lambda t: pd.to_datetime(t, unit="s"))
            df["Year"], df["Month"], df["Day"] = df["Time"].dt.year, df["Time"].dt.month, df["Time"].dt.day
            df.to_parquet(bronze_path, partition_cols=["Year", "Month", "Day"], index=False)
            tmp_flag = success_flag + ".tmp"
            with open(tmp_flag, "w") as f:
                f.write("SUCCESS")
            os.replace(tmp_flag, success_flag)
        except Exception as e:
            if os.path.exists(bronze_path):
                os.remove(bronze_path)
            if os.path.exists(success_flag):
                os.remove(success_flag)
            if os.path.exists(success_flag + ".tmp"):
                os.remove(success_flag + ".tmp")
            raise e
        return bronze_path

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def bronze_to_silver(bronze_path: str, ds: str = None) -> str:
        from airflow.utils.log.logging_mixin import LoggingMixin
        log = LoggingMixin().log
        from src.amazon_recommender.data import load_and_clean
        from src.amazon_recommender.features import filter_users_items
        from src.amazon_recommender.model import build_sparse_matrix
        from scipy.sparse import save_npz

        silver_root = Variable.get("silver_root")
        silver_dir = os.path.join(silver_root, ds)
        os.makedirs(silver_dir, exist_ok=True)
        silver_path = os.path.join(silver_dir, "train.npz")
        success_flag = os.path.join(silver_dir, "_SUCCESS")
        if os.path.exists(success_flag) and os.path.getsize(success_flag) > 0:
            return silver_path

        if os.path.exists(silver_path):
            log.info(f"Silver artifact already exists at {silver_path}, skipping.")
            return silver_path

        try:
            df = pd.read_parquet(bronze_path)
            cleaned = load_and_clean(df)
            filtered = filter_users_items(cleaned)
            M, ui, ii = build_sparse_matrix(filtered)
            save_npz(silver_path, M)
            tmp_flag = success_flag + ".tmp"
            with open(tmp_flag, "w") as f:
                f.write("SUCCESS")
            os.replace(tmp_flag, success_flag)
            log.info(f"Saved sparse matrix to {silver_path}")
        except Exception as e:
            if os.path.exists(silver_path):
                os.remove(silver_path)
            if os.path.exists(success_flag):
                os.remove(success_flag)
            if os.path.exists(success_flag + ".tmp"):
                os.remove(success_flag + ".tmp")
            raise e
        return silver_path

    @task(pool="training_pool", sla=timedelta(hours=1))
    def train_models(silver_path: str) -> dict:
        from airflow.utils.log.logging_mixin import LoggingMixin
        log = LoggingMixin().log
        from src.amazon_recommender.model import build_recommender
        import pickle
        import mlflow
        import os

        # Train models
        svd_model, knn_model, train_matrix = build_recommender(silver_path)

        # Save pickles
        artifacts_dir = Variable.get("artifacts_dir", default_var="artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        svd_path = os.path.join(artifacts_dir, "svd_model.pkl")
        knn_path = os.path.join(artifacts_dir, "knn_model.pkl")
        matrix_path = os.path.join(artifacts_dir, "train_matrix.pkl")
        with open(svd_path, "wb") as f:
            pickle.dump(svd_model, f)
        with open(knn_path, "wb") as f:
            pickle.dump(knn_model, f)
        with open(matrix_path, "wb") as f:
            pickle.dump(train_matrix, f)
        log.info(f"Saved SVD, KNN, and train matrix pickles to {artifacts_dir}")

        # Log to MLflow
        mlflow_conn_id = Variable.get("mlflow_conn_id", default_var=None)
        mlflow_hook = MlflowHook(mlflow_conn_id or "mlflow_default")
        mlflow = mlflow_hook.get_client()
        with mlflow.start_run(run_name="recommender_training"):
            mlflow.log_artifact(svd_path)
            mlflow.log_artifact(knn_path)
            mlflow.log_artifact(matrix_path)
            log.info("Logged artifacts to MLflow")

        return {"svd": svd_path, "knn": knn_path, "matrix": matrix_path}

    @task(sla=timedelta(minutes=30))
    def drift_report(silver_path: str) -> float:
        from airflow.utils.log.logging_mixin import LoggingMixin
        log = LoggingMixin().log
        from monitoring.drift_report import generate_drift_report
        import os
        from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

        # Generate drift report and score
        drift_score, drift_html = generate_drift_report(silver_path)

        # Save HTML report
        report_dir = Variable.get("drift_report_dir", default_var="artifacts")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "drift_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(drift_html)
        log.info(f"Saved drift report to {report_path}")

        # Log to MLflow
        mlflow_conn_id = Variable.get("mlflow_conn_id", default_var=None)
        mlflow_hook = MlflowHook(mlflow_conn_id or "mlflow_default")
        mlflow = mlflow_hook.get_client()
        with mlflow.start_run(run_name="drift_report"):
            mlflow.log_artifact(report_path)
            mlflow.log_metric("drift_score", drift_score)
            log.info("Logged drift report and metric to MLflow")

        drift_thr = float(Variable.get("drift_threshold"))
        if drift_score > drift_thr:
            slack_conn_id = Variable.get("slack_conn_id", default_var=None)
            if slack_conn_id:
                SlackWebhookOperator(
                    task_id="slack_alert",
                    http_conn_id=slack_conn_id,
                    message=f":warning: Drift score {drift_score:.3f} exceeded threshold!",
                ).execute(context={})

        return drift_score

    @task.branch()
    def decide_branch(drift_score: float, ds: str = None) -> str:
        from airflow.utils.log.logging_mixin import LoggingMixin
        log = LoggingMixin().log
        drift_thr = float(Variable.get("drift_threshold"))
        import datetime as dt
        is_weekly = dt.datetime.strptime(ds, "%Y-%m-%d").weekday() == 6
        if drift_score > drift_thr or is_weekly:
            log.info("Retraining triggered.")
            return "train_models"
        log.info("No retraining needed.")
        return "end"



    @task(trigger_rule="none_failed_or_skipped")
    def housekeeping(ds=None):
        import shutil
        import glob
        import os
        from datetime import datetime, timedelta
        from airflow.utils.log.logging_mixin import LoggingMixin
        log = LoggingMixin().log
        retention_days = int(Variable.get("retention_days", default_var=30))
        bronze_root = Variable.get("bronze_root")
        silver_root = Variable.get("silver_root")
        cutoff = datetime.strptime(ds, "%Y-%m-%d") - timedelta(days=retention_days)
        # Prune bronze
        for d in os.listdir(bronze_root):
            try:
                d_path = os.path.join(bronze_root, d)
                d_date = datetime.strptime(d, "%Y-%m-%d")
                if d_date < cutoff:
                    shutil.rmtree(d_path)
                    log.info(f"Pruned bronze partition {d_path}")
            except Exception:
                continue
        # Prune silver
        for d in os.listdir(silver_root):
            try:
                d_path = os.path.join(silver_root, d)
                d_date = datetime.strptime(d, "%Y-%m-%d")
                if d_date < cutoff:
                    shutil.rmtree(d_path)
                    log.info(f"Pruned silver partition {d_path}")
            except Exception:
                continue
        return True

    @task()
    def maybe_alert(drift_score: float):
        drift_thr = float(Variable.get("drift_threshold"))
        if drift_score > drift_thr:
            return "alert"
        return "no_alert"

    from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
    slack_conn_id = Variable.get("slack_conn_id", default_var=None)
    alert = SlackWebhookOperator(
        task_id="slack_alert",
        http_conn_id=slack_conn_id,
        message=":warning: Drift score exceeded threshold!",
    ) if slack_conn_id else None

    end = EmptyOperator(task_id="end")



    bronze = raw_to_bronze()
    silver = bronze_to_silver(bronze)
    drift = drift_report(silver)
    branch = decide_branch(drift)
    trained = train_models(silver)
    house = housekeeping()

    bronze >> silver >> drift >> branch
    drift >> maybe_alert(drift) >> alert if alert else drift
    branch >> trained >> house
    branch >> end >> house

recommender_dag = recommender_pipeline()
