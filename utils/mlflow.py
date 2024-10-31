from __future__ import annotations

import os.path as osp
import shutil
from typing import Any

import mlflow


MLRUNS_PATH = "./mlruns"


class MyMLflow:
    @staticmethod
    def set_exp(exp_name: str, mlflow_port: int, overwrite: bool = False) -> str:
        """
        Create mlflow experiment

        Arguments:
            exp_name: str of the experiment name
            mlflow_port: port for mlflow connection
            overwrite: True to overwrite previous runs
        """
        mlflow.set_tracking_uri(uri=f"http://127.0.0.1:{mlflow_port}")
        if experiment := mlflow.get_experiment_by_name(exp_name):
            if overwrite:  # ! DEPRECATED: never enters here
                mlflow.delete_experiment(experiment.experiment_id)
                shutil.rmtree(osp.join(MLRUNS_PATH, ".trash", experiment.experiment_id))
                exp_id = mlflow.create_experiment(exp_name)
            else:
                exp_id = experiment.experiment_id
        else:
            exp_id = mlflow.create_experiment(exp_name)
        mlflow.set_experiment(experiment_id=exp_id)
        return exp_id

    @staticmethod
    def get_run(exp_id: str, run_name: str) -> str | None:
        """
        Get mlflow run_id with exp_id and run_name
        """
        filter_string = f"tags.mlflow.runName = '{run_name}'"
        runs = mlflow.search_runs(exp_id, filter_string)
        run_id = runs["run_id"].values[0] if len(runs) > 0 else None
        return run_id

    @staticmethod
    def del_run(exp_id: str, run_name: str) -> None:
        """
        Delete mlflow run with exp_id and run_name
        """
        run_id = MyMLflow.get_run(exp_id, run_name)
        if run_id is not None:
            mlflow.delete_run(run_id)

    @staticmethod
    def start_run(
        exp_id: str,
        run_name: str,
        father: bool = False,
        overwrite: bool = False,
    ) -> None:
        run_id = MyMLflow.get_run(exp_id, run_name)
        if run_id is not None and father and overwrite:
            input(
                "A run with the same name has been found! You are about to "
                "delete this run and ALL of it's children! Press Enter to "
                "continue or CTRL+C to stop!"
            )
            children_run_id = mlflow.search_runs(
                exp_id, f"tags.mlflow.parentRunId = '{run_id}'"
            )["run_id"].values.tolist()
            for child_run_id in children_run_id:
                mlflow.delete_run(child_run_id)
            mlflow.delete_run(run_id)
            run_id = None
        mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name,
            run_id=run_id,
            nested=True,
        )

    @staticmethod
    def end_run() -> None:
        mlflow.end_run()

    @staticmethod
    def set_tag(key: str, value: Any) -> None:  # noqa: ANN401
        mlflow.set_tag(key, value)

    @staticmethod
    def log_params(params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    @staticmethod
    def log_metrics(metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    @staticmethod
    def log_artifacts(artifacts: dict[str, str]) -> None:
        for artifact_path, local_path in artifacts.items():
            mlflow.log_artifact(local_path, artifact_path)
