"""
Log runs to Weights and Biases.
"""

import os
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from wandb.apis.importers.mlflow import MlflowImporter
from wandb.apis.importers import Namespace


WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def has_useful_runs(client: MlflowClient, exp_id: str) -> bool:
    """Check if an Mlflow client contains any finished (non-empty) experiments""" 
    runs = client.search_runs(
        [exp_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1,
    )
    if not runs:
        return False
    r = runs[0]
    return bool(r.data.metrics or r.data.params or r.data.tags)


def get_nonempty_experiments(track_uri: str):
    """Get all non-empty experiments under the specified URI"""
    client = MlflowClient(tracking_uri=track_uri)
    good_experiments = []
    for exp in client.search_experiments():
        try:
            if not has_useful_runs(client, exp.experiment_id):
                continue
            good_experiments.append(exp)
        except MlflowException as e:
            print(f"[skip exp {exp.experiment_id}] {e}")
            continue
    return good_experiments


def main() -> None:
    """Log Auto3DSeg runs in the MLFLow format to W&B"""
    work_dir = Path("/home/renku/work/auto3dseg-runs/run-002")
    mlruns = [f"file://{str(dir)}" for dir in work_dir.rglob("*mlruns/")]
    for mlflow_dir in mlruns:
        importer = MlflowImporter(
            mlflow_tracking_uri=mlruns[0],
            dst_base_url="https://api.wandb.ai",
            dst_api_key=WANDB_API_KEY,
        )
        runs = importer.collect_runs()
        ns = Namespace(entity="dbottone-ethz", project="ISLES")
        importer.import_runs(runs, namespace=ns)


if __name__ == "__main__":
    main()
