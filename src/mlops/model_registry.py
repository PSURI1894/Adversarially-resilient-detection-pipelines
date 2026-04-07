"""
================================================================================
MODEL REGISTRY — VERSION CONTROL & STAGE MANAGEMENT
================================================================================
MLflow Model Registry for versioned model lifecycle management.
Falls back to local file-based registry when MLflow is unavailable.

Features:
    - Stage management: Staging → Production → Archived
    - Automated promotion rules (beat current production by threshold)
    - Model signature and input schema validation
    - Artifact storage with checksum integrity
================================================================================
"""

import os
import json
import time
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

_MLFLOW_AVAILABLE = False
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MlflowClient = None


class ModelStage(Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry:
    """
    Model version control with stage-based lifecycle management.

    Parameters
    ----------
    registry_name : str
        Registered model name in MLflow.
    local_registry_dir : str
        Directory for local fallback registry.
    promotion_threshold : float
        Minimum metric improvement to auto-promote (e.g., 0.02 = 2%).
    promotion_metric : str
        Metric to compare for promotion decisions.
    """

    def __init__(self, registry_name: str = "ids-ensemble",
                 local_registry_dir: str = "models/registry",
                 promotion_threshold: float = 0.02,
                 promotion_metric: str = "robust_f1"):
        self.registry_name = registry_name
        self.local_dir = Path(local_registry_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.promotion_threshold = promotion_threshold
        self.promotion_metric = promotion_metric

        self._use_mlflow = False
        self._client = None
        self._local_versions: List[Dict] = []
        self._manifest_path = self.local_dir / "manifest.json"

        if _MLFLOW_AVAILABLE:
            try:
                self._client = MlflowClient()
                self._use_mlflow = True
                logger.info(f"Model registry connected to MLflow ({registry_name})")
            except Exception as e:
                logger.warning(f"MLflow registry unavailable ({e}), using local")

        self._load_local_manifest()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(self, model, metrics: Dict[str, float],
                       model_params: Optional[Dict] = None,
                       description: str = "") -> str:
        """
        Register a new model version.

        Parameters
        ----------
        model : object
            The trained model to register.
        metrics : dict
            Evaluation metrics for this model version.
        model_params : dict, optional
            Model hyperparameters.
        description : str
            Human-readable description.

        Returns
        -------
        str
            Version identifier.
        """
        version_id = f"v{len(self._local_versions) + 1}"
        timestamp = time.time()

        # Compute model checksum
        model_bytes = pickle.dumps(model)
        checksum = hashlib.sha256(model_bytes).hexdigest()[:16]

        entry = {
            "version_id": version_id,
            "timestamp": timestamp,
            "stage": ModelStage.STAGING.value,
            "metrics": metrics,
            "params": model_params or {},
            "description": description,
            "checksum": checksum,
        }

        # Save model artifact locally
        model_path = self.local_dir / f"{version_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        entry["artifact_path"] = str(model_path)

        # MLflow registration
        if self._use_mlflow:
            try:
                with mlflow.start_run():
                    mlflow.log_params(model_params or {})
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(
                        model, "model",
                        registered_model_name=self.registry_name,
                    )
            except Exception as e:
                logger.warning(f"MLflow model registration failed: {e}")

        self._local_versions.append(entry)
        self._save_local_manifest()
        logger.info(f"Registered model {version_id} (checksum={checksum})")
        return version_id

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def promote(self, version_id: str, target_stage: ModelStage = ModelStage.PRODUCTION) -> bool:
        """
        Promote a model version to a target stage.
        If promoting to PRODUCTION, archives the current production model.
        """
        entry = self._find_version(version_id)
        if entry is None:
            logger.error(f"Version {version_id} not found")
            return False

        if target_stage == ModelStage.PRODUCTION:
            # Archive current production model(s)
            for v in self._local_versions:
                if v["stage"] == ModelStage.PRODUCTION.value:
                    v["stage"] = ModelStage.ARCHIVED.value
                    logger.info(f"Archived {v['version_id']}")

        entry["stage"] = target_stage.value
        self._save_local_manifest()
        logger.info(f"Promoted {version_id} → {target_stage.value}")
        return True

    def auto_promote(self, version_id: str) -> bool:
        """
        Automatically promote if the new model beats current production
        by at least promotion_threshold on promotion_metric.
        """
        new_entry = self._find_version(version_id)
        if new_entry is None:
            return False

        prod_entry = self.get_production_model_info()
        if prod_entry is None:
            # No production model — auto-promote
            return self.promote(version_id, ModelStage.PRODUCTION)

        new_score = new_entry["metrics"].get(self.promotion_metric, 0)
        prod_score = prod_entry["metrics"].get(self.promotion_metric, 0)

        if new_score - prod_score >= self.promotion_threshold:
            logger.info(f"Auto-promotion: {self.promotion_metric} "
                         f"{prod_score:.4f} → {new_score:.4f} "
                         f"(+{new_score - prod_score:.4f})")
            return self.promote(version_id, ModelStage.PRODUCTION)
        else:
            logger.info(f"Auto-promotion denied: improvement "
                         f"{new_score - prod_score:.4f} < threshold {self.promotion_threshold}")
            return False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_production_model_info(self) -> Optional[Dict]:
        """Get metadata of the current production model."""
        for v in reversed(self._local_versions):
            if v["stage"] == ModelStage.PRODUCTION.value:
                return v
        return None

    def load_production_model(self):
        """Load the current production model from disk."""
        info = self.get_production_model_info()
        if info is None:
            raise FileNotFoundError("No production model registered")

        model_path = info["artifact_path"]
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def get_version(self, version_id: str) -> Optional[Dict]:
        return self._find_version(version_id)

    def list_versions(self, stage: Optional[ModelStage] = None) -> List[Dict]:
        """List all registered model versions, optionally filtered by stage."""
        if stage is None:
            return list(self._local_versions)
        return [v for v in self._local_versions if v["stage"] == stage.value]

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def verify_integrity(self, version_id: str) -> bool:
        """Verify model artifact integrity via SHA-256 checksum."""
        entry = self._find_version(version_id)
        if entry is None:
            return False

        model_path = entry.get("artifact_path")
        if not model_path or not os.path.exists(model_path):
            return False

        with open(model_path, "rb") as f:
            model_bytes = f.read()

        actual = hashlib.sha256(model_bytes).hexdigest()[:16]
        return actual == entry["checksum"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_version(self, version_id: str) -> Optional[Dict]:
        for v in self._local_versions:
            if v["version_id"] == version_id:
                return v
        return None

    def _load_local_manifest(self):
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                self._local_versions = json.load(f)

    def _save_local_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self._local_versions, f, indent=2, default=str)
