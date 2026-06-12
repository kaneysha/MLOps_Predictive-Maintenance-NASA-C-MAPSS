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


def fetch_latest_approved_run(exp_name: str):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        logging.error(f"Experiment not found: {exp_name}")
        return None

    try:
        results = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.model_status = 'accepted'",
            order_by=["start_time DESC"],
            max_results=1
        )

        if len(results) == 0:
            logging.warning("Tidak ada run dengan status accepted")
            return None

        latest_run = results[0]
        logging.info(f"Run ditemukan: {latest_run.info.run_id}")
        return latest_run

    except Exception as e:
        logging.error(f"Gagal mengambil run: {str(e)}")
        return None


def push_to_registry(run_id: str, model_id: str, exp_name: str, stage_name: str = "Staging"):

    client = MlflowClient()
    model_source = f"runs:/{run_id}/model"

    logging.info(f"Model source URI: {model_source}")

    try:
        model_version = mlflow.register_model(
            model_uri=model_source,
            name=model_id,
            tags={
                "experiment_name": exp_name,
                "origin_run": run_id,
                "auto_push": "yes"
            }
        )

        logging.info(f"Model registered -> {model_version.name} v{model_version.version}")

        if stage_name and stage_name.lower() != "none":
            client.transition_model_version_stage(
                name=model_id,
                version=model_version.version,
                stage=stage_name,
                archive_existing_versions=True
            )
            logging.info(f"Stage updated ke: {stage_name}")

        return True

    except Exception as e:
        logging.error(f"Register model gagal: {str(e)}")
        return False


def run_pipeline():

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "nasa_cmapss")
    model_id = os.getenv("MLFLOW_MODEL_NAME", "nasa_cmapss_model")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    mlflow.set_tracking_uri(tracking_uri)

    logging.info("=== Model Registry Pipeline Started ===")
    logging.info(f"Experiment : {exp_name}")
    logging.info(f"Model Name : {model_id}")
    logging.info(f"Tracking   : {tracking_uri}")

    run = fetch_latest_approved_run(exp_name)

    if run is None:
        logging.error("Tidak ada run valid untuk diregister")
        return False

    success = push_to_registry(
        run_id=run.info.run_id,
        model_id=model_id,
        exp_name=exp_name,
        stage_name="Staging"
    )

    if success:
        logging.info("Pipeline selesai dengan sukses")
        return True

    logging.error("Pipeline gagal saat registrasi model")
    return False


if __name__ == "__main__":
    ok = run_pipeline()
    sys.exit(0 if ok else 1)