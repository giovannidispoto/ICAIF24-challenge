import os
import mlflow
import optuna

from optuna.samplers import TPESampler

from task1_eval import run_evaluation
from utils.mlflow import MyMLflow


from dotenv import load_dotenv

load_dotenv()

PROJECT_FOLDER = os.getenv("PROJECT_FOLDER")
AGENTS_FOLDER = os.path.join(PROJECT_FOLDER, "agents")
RUNS_FOLDER = os.path.join(PROJECT_FOLDER, "runs")


def sample_params(trial: optuna.Trial) -> dict:
    agents_weights_upd_freq = trial.suggest_int("agents_weights_upd_freq", 1, 10)
    loss_fn_window = trial.suggest_int("loss_fn_window", 1, 10)
    action_thresh = trial.suggest_float("action_thresh", 0.5, 1, step=0.05)
    return {
        "agents_weights_upd_freq": agents_weights_upd_freq,
        "loss_fn_window": loss_fn_window,
        "action_thresh": action_thresh,
    }


def objective(trial: optuna.Trial) -> float:
    run_name = f"{RUN_NAME}_trial_{trial.number}"
    run_args = sample_params(trial)
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(run_args)
        value = run_evaluation(
            run_name=os.path.join(RUN_NAME, run_name),
            agents_names=AGENTS_NAMES,
            oamp_args=sample_params(trial),
        )
        mlflow.log_artifact(
            local_path=os.path.join(RUNS_FOLDER, RUN_NAME, run_name, "oamp_stats.png"),
            artifact_path="oamp_stats",
        )
        return value


if __name__ == "__main__":
    EXP_NAME = "oamp_tuning"
    RUN_NAME = "exp_0"
    MLFLOW_PORT = 5000

    AGENTS_NAMES = ["agent_0", "agent_1", "agent_2", "agent_3"]
    N_TRIALS = 100
    N_STARTUP_TRIALS = 10

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)

    exp_id = MyMLflow.set_exp(EXP_NAME, MLFLOW_PORT)

    MyMLflow.start_run(exp_id=exp_id, run_name=RUN_NAME, overwrite=True)

    study = optuna.create_study(
        study_name=RUN_NAME,
        sampler=TPESampler(n_startup_trials=N_STARTUP_TRIALS),
        direction="maximize",
        # storage="mysql://pfm:password@localhost/OptunaPPO",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_value", study.best_value)
    mlflow.log_metric("best_trial", study.best_trial)
    mlflow.log_param("n_trials", len(study.trials))
