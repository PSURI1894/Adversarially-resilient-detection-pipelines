"""
================================================================================
EXPERIMENT TRACKER — MLFLOW INTEGRATION
================================================================================
Logs hyperparameters, metrics, artifacts, and models to MLflow.
Falls back to local file-based tracking when MLflow server is unavailable.

Features:
    - Auto-logging of losses, F1, FDR, AUC, ECE, RSCP coverage, set sizes
    - Comparison dashboard generation
    - Run tagging (attack type, defense type, dataset version)
    - Graceful fallback to local JSON logging
================================================================================
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None


class ExperimentTracker:
    """
    Unified experiment tracker with MLflow backend + local fallback.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name.
    tracking_uri : str or None
        MLflow tracking server URI. None → local ./mlruns.
    fallback_dir : str
        Directory for local JSON logs when MLflow is unavailable.
    """

    def __init__(self, experiment_name: str = "ids-pipeline",
                 tracking_uri: Optional[str] = None,
                 fallback_dir: str = "reports/experiments"):
        self.experiment_name = experiment_name
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

        self._use_mlflow = False
        self._client = None
        self._active_run = None
        self._local_runs: List[Dict] = []

        if _MLFLOW_AVAILABLE:
            try:
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                self._client = MlflowClient()
                self._use_mlflow = True
                logger.info(f"MLflow tracker initialised (experiment={experiment_name})")
            except Exception as e:
                logger.warning(f"MLflow unavailable ({e}), using local fallback")

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new experiment run. Returns run ID."""
        if self._use_mlflow:
            self._active_run = mlflow.start_run(run_name=run_name)
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))
            run_id = self._active_run.info.run_id
        else:
            run_id = f"local-{int(time.time())}"
            self._active_run = {
                "run_id": run_id,
                "run_name": run_name,
                "tags": tags or {},
                "params": {},
                "metrics": {},
                "artifacts": [],
                "start_time": time.time(),
            }
        logger.info(f"Started run: {run_id}")
        return run_id

    def end_run(self):
        """End the current run."""
        if self._use_mlflow and self._active_run:
            mlflow.end_run()
        elif self._active_run and isinstance(self._active_run, dict):
            self._active_run["end_time"] = time.time()
            self._local_runs.append(self._active_run)
            self._save_local_run(self._active_run)
        self._active_run = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        if self._use_mlflow:
            mlflow.log_params({k: str(v) for k, v in params.items()})
        elif self._active_run:
            self._active_run["params"].update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if self._use_mlflow:
            mlflow.log_metric(key, value, step=step)
        elif self._active_run:
            if key not in self._active_run["metrics"]:
                self._active_run["metrics"][key] = []
            self._active_run["metrics"][key].append({
                "value": value, "step": step, "timestamp": time.time()
            })

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        if self._use_mlflow:
            mlflow.log_artifact(local_path, artifact_path)
        elif self._active_run:
            self._active_run["artifacts"].append(local_path)

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        if self._use_mlflow:
            mlflow.set_tag(key, value)
        elif self._active_run:
            self._active_run["tags"][key] = value

    # ------------------------------------------------------------------
    # Pipeline-specific auto-logging
    # ------------------------------------------------------------------

    def log_pipeline_stage(self, stage: str, metrics: Dict[str, float],
                           params: Optional[Dict[str, Any]] = None):
        """
        Convenience method for logging a complete pipeline stage.

        Parameters
        ----------
        stage : str
            Stage name (e.g., 'training', 'conformal_calibration', 'evaluation').
        metrics : dict
            Stage metrics (F1, FDR, AUC, ECE, coverage, set_size, etc.).
        params : dict, optional
            Stage-specific parameters.
        """
        prefixed_metrics = {f"{stage}/{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed_metrics)

        if params:
            prefixed_params = {f"{stage}/{k}": v for k, v in params.items()}
            self.log_params(prefixed_params)

        self.set_tag(f"stage_{stage}", "completed")

    def log_attack_result(self, attack_type: str, epsilon: float,
                          metrics: Dict[str, float]):
        """Log results from an adversarial attack evaluation."""
        self.set_tag("attack_type", attack_type)
        self.log_metric("attack/epsilon", epsilon)
        for k, v in metrics.items():
            self.log_metric(f"attack/{k}", v)

    # ------------------------------------------------------------------
    # Model logging
    # ------------------------------------------------------------------

    def log_model(self, model, artifact_path: str = "model",
                  registered_name: Optional[str] = None):
        """Log a model to the tracking backend."""
        if self._use_mlflow:
            try:
                mlflow.sklearn.log_model(
                    model, artifact_path,
                    registered_model_name=registered_name,
                )
            except Exception:
                # Fallback: try generic pyfunc logging
                try:
                    mlflow.pyfunc.log_model(artifact_path, python_model=model)
                except Exception as e:
                    logger.warning(f"Model logging failed: {e}")
        else:
            logger.info(f"Model logged locally (artifact_path={artifact_path})")

    # ------------------------------------------------------------------
    # Comparison & retrieval
    # ------------------------------------------------------------------

    def get_best_run(self, metric: str = "evaluation/f1",
                     ascending: bool = False) -> Optional[Dict]:
        """Get the best run by a given metric."""
        if self._use_mlflow:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None
            order = "ASC" if ascending else "DESC"
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.`{metric}` {order}"],
                max_results=1,
            )
            if len(runs) > 0:
                return runs.iloc[0].to_dict()
            return None
        else:
            # Local: search through saved runs
            if not self._local_runs:
                return None
            best = None
            best_val = float("-inf") if not ascending else float("inf")
            for run in self._local_runs:
                vals = run.get("metrics", {}).get(metric, [])
                if vals:
                    v = vals[-1]["value"]
                    if (not ascending and v > best_val) or (ascending and v < best_val):
                        best_val = v
                        best = run
            return best

    # ------------------------------------------------------------------
    # Local persistence
    # ------------------------------------------------------------------

    def _save_local_run(self, run: Dict):
        """Persist a local run to JSON."""
        run_file = self.fallback_dir / f"{run['run_id']}.json"
        # Convert non-serializable types
        serializable = json.loads(json.dumps(run, default=str))
        with open(run_file, "w") as f:
            json.dump(serializable, f, indent=2)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._active_run:
            self.end_run()
        return False
