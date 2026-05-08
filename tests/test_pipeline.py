from src.pipeline import run_pipeline

def test_pipeline_runs_successfully():
    result = run_pipeline()

    # pipeline harus selesai tanpa error
    assert result is not None