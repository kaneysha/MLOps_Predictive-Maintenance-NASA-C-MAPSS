#!/usr/bin/env python3
import os
import sys
import logging
from dotenv import load_dotenv

import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def fetch_latest_metrics(experiment_name: str):

    client = MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logging.error(f"Experiment '{experiment_name}' tidak ditemukan")
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )

        if len(runs) == 0:
            logging.error("Belum ada training run")
            return None

        latest_run = runs[0]

        logging.info(f"Latest run id: {latest_run.info.run_id}")

        return latest_run.data.metrics

    except Exception as err:
        logging.error(f"Gagal mengambil metrics: {err}")
        return None


def fetch_current_production(model_name: str):

    client = MlflowClient()

    try:
        prod_models = client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        if not prod_models:
            logging.warning("Belum ada model Production")
            return None

        prod_run_id = prod_models[0].run_id
        prod_run = client.get_run(prod_run_id)

        logging.info(
            f"Production run ditemukan: {prod_run_id[:8]}..."
        )

        return prod_run.data.metrics

    except Exception as err:
        logging.warning(f"Tidak bisa membaca Production model: {err}")
        return None


def evaluate_candidate(
    candidate_metrics: dict,
    production_metrics: dict | None,
    min_r2: float,
    max_rmse: float
):
    """Evaluasi model baru."""

    candidate_r2 = candidate_metrics.get("R2", 0.0)
    candidate_rmse = candidate_metrics.get("RMSE", 9999.0)

    logging.info("=== Candidate Model Metrics ===")
    logging.info(f"R2   : {candidate_r2:.4f}")
    logging.info(f"RMSE : {candidate_rmse:.4f}")

    # threshold validation
    if candidate_r2 < min_r2:
        logging.error(
            f"R2 terlalu rendah ({candidate_r2:.4f} < {min_r2:.4f})"
        )
        return False

    if candidate_rmse > max_rmse:
        logging.error(
            f"RMSE terlalu tinggi ({candidate_rmse:.4f} > {max_rmse:.4f})"
        )
        return False

    logging.info("Lolos minimum quality gate")

    # compare with production
    if production_metrics:

        prod_r2 = production_metrics.get("R2", 0.0)
        prod_rmse = production_metrics.get("RMSE", 9999.0)

        logging.info("=== Production Metrics ===")
        logging.info(f"R2   : {prod_r2:.4f}")
        logging.info(f"RMSE : {prod_rmse:.4f}")

        if candidate_r2 >= prod_r2:
            logging.info("Candidate punya R2 lebih baik/equal")

        if candidate_rmse <= prod_rmse:
            logging.info("Candidate punya RMSE lebih baik/equal")

        if candidate_r2 < prod_r2:
            logging.warning("R2 model baru lebih rendah dari Production")

        if candidate_rmse > prod_rmse:
            logging.warning("RMSE model baru lebih buruk dari Production")

    else:
        logging.info("Tidak ada Production model untuk dibandingkan")

    return True


def main():

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "nasa_cmapss"
    )

    model_name = os.getenv(
        "MLFLOW_MODEL_NAME",
        "nasa_cmapss_model"
    )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db"
    )

    min_r2 = float(os.getenv("MIN_R2", "0.50"))
    max_rmse = float(os.getenv("MAX_RMSE", "20"))

    mlflow.set_tracking_uri(tracking_uri)

    logging.info("Memulai evaluasi model...")
    logging.info(f"Experiment : {experiment_name}")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logging.error(f"Experiment '{experiment_name}' tidak ditemukan")
        sys.exit(1)

    latest_metrics = fetch_latest_metrics(experiment_name)

    if latest_metrics is None:
        sys.exit(1)

    production_metrics = fetch_current_production(model_name)

    passed = evaluate_candidate(
        candidate_metrics=latest_metrics,
        production_metrics=production_metrics,
        min_r2=min_r2,
        max_rmse=max_rmse
    )

    logging.info("=== Final Decision ===")

    if passed:
        # Tag the latest run as accepted for registration
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if runs:
            latest_run_id = runs[0].info.run_id
            client.set_tag(latest_run_id, "model_status", "accepted")
            logging.info(f"Run tagged: {latest_run_id} → model_status=accepted")
            logging.info("Model layak masuk staging")
            sys.exit(0)

    logging.error("Model ditolak")
    sys.exit(1)


if __name__ == "__main__":
    main()