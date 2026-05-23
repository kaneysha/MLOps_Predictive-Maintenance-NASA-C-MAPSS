"""Helpers to configure MLflow to use MinIO (S3-compatible) as an artifact store.

This module sets the necessary environment variables for boto3/MLflow and
optionally sets the MLflow tracking URI.
"""
from typing import Optional
import os
import mlflow


def configure_mlflow(
    tracking_uri: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    aws_access_key: Optional[str] = None,
    aws_secret_key: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tracking_username: Optional[str] = None,
    tracking_password: Optional[str] = None,
):
    """Configure MLflow and environment for S3-compatible artifact storage.

    - Sets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` when provided.
    - Sets `MLFLOW_S3_ENDPOINT_URL` to point MLflow/boto3 to the MinIO endpoint.
    - Calls `mlflow.set_tracking_uri()` when `tracking_uri` is provided.
    - Calls `mlflow.set_experiment()` when `experiment_name` is provided.

    These environment variables make MLflow use boto3 to upload artifacts to
    MinIO (or any S3-compatible service).
    """

    if aws_access_key:
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
    if aws_secret_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_key
    if s3_endpoint:
        # Tell MLflow/boto3 to use a custom S3 endpoint (MinIO)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint

    # Optional: basic auth for tracking server (MLflow client will use requests)
    if tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
    if tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password

    # Optional: set tracking uri
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # Keep existing default if not provided
        mlflow.set_tracking_uri("file:./mlruns")

    if experiment_name:
        mlflow.set_experiment(experiment_name)
