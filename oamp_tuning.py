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
os.makedirs(AGENTS_FOLDER, exist_ok=True)
RUNS_FOLDER = os.path.join(PROJECT_FOLDER, "runs")
os.makedirs(RUNS_FOLDER, exist_ok=True)


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

    # to launch mlflow dashboard
    # mlflow server --backend-store-uri file:/home/trading/antonio/ICAIF24-challenge/mlruns --port=5005

    # to launch optuna dashboard
    # optuna-dashboard mysql+pymysql://ICAIF@localhost/OptunaOAMP

    ## Experiment Params
    EXP_NAME = "oamp_tuning"                                        
    RUN_NAME = "exp_0"
    AGENTS_NAMES = ["agent_0", "agent_1", "agent_2", "agent_3"]

    ## MLflow Params
    MLFLOW_PORT = 5005
    # Creating new MLflow run
    exp_id = MyMLflow.set_exp(EXP_NAME, MLFLOW_PORT)
    MyMLflow.start_run(exp_id=exp_id, run_name=RUN_NAME, father=True, overwrite=True)

    ## Optuna Params
    STUDY_NAME = RUN_NAME
    N_TRIALS = 5
    N_STARTUP_TRIALS = 2
    SAMPLER = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    STORAGE = "mysql+pymysql://ICAIF@localhost/OptunaOAMP"
    OVERWRITE = True
    # Creating new Optuna study
    if OVERWRITE:
       study_names = optuna.study.get_all_study_names(STORAGE)
       if STUDY_NAME in study_names:
            optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE)
    study = optuna.create_study(
        study_name=RUN_NAME,
        sampler=SAMPLER,
        direction="maximize",
        storage=STORAGE,
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_value", study.best_value)
    mlflow.log_metric("best_trial", study.best_trial.number)
    mlflow.log_param("n_trials", len(study.trials))
