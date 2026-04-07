"""
================================================================================
DATA VERSIONING — DATASET LINEAGE & INTEGRITY TRACKING
================================================================================
DVC-like data versioning for datasets and processed artifacts.

Features:
    - SHA-256 checksums for data integrity verification
    - Lineage tracking: raw → processed → train/cal/test splits
    - Reproducibility: any experiment can be recreated from version hash
    - Lightweight: no external service dependencies
================================================================================
"""

import json
import time
import hashlib
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


class DataVersioner:
    """
    Lightweight data versioning with lineage tracking and integrity checks.

    Parameters
    ----------
    registry_dir : str
        Directory storing version metadata.
    """

    def __init__(self, registry_dir: str = "data/.versions"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.registry_dir / "manifest.json"
        self._manifest: Dict[str, Dict] = {}
        self._load_manifest()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_file(
        self,
        file_path: str,
        description: str = "",
        parent_version: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Register a data file and compute its version hash.

        Parameters
        ----------
        file_path : str
            Path to the data file.
        description : str
            Human-readable description.
        parent_version : str, optional
            Version hash of the parent dataset (for lineage).
        metadata : dict, optional
            Additional metadata (split ratios, column names, etc.).

        Returns
        -------
        str
            Version hash (SHA-256 of file contents).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        checksum = self._compute_checksum(file_path)
        file_size = file_path.stat().st_size

        entry = {
            "file_path": str(file_path.resolve()),
            "file_name": file_path.name,
            "checksum": checksum,
            "file_size": file_size,
            "description": description,
            "parent_version": parent_version,
            "metadata": metadata or {},
            "registered_at": time.time(),
        }

        self._manifest[checksum] = entry
        self._save_manifest()
        logger.info(f"Registered data: {file_path.name} → {checksum[:12]}...")
        return checksum

    def register_array(
        self,
        array: np.ndarray,
        name: str,
        description: str = "",
        parent_version: Optional[str] = None,
    ) -> str:
        """
        Register an in-memory numpy array (e.g., a train/test split).
        Computes hash over array bytes.
        """
        array_bytes = array.tobytes()
        checksum = hashlib.sha256(array_bytes).hexdigest()

        entry = {
            "file_path": f"<in-memory:{name}>",
            "file_name": name,
            "checksum": checksum,
            "file_size": len(array_bytes),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "description": description,
            "parent_version": parent_version,
            "metadata": {},
            "registered_at": time.time(),
        }

        self._manifest[checksum] = entry
        self._save_manifest()
        logger.info(f"Registered array: {name} {array.shape} → {checksum[:12]}...")
        return checksum

    # ------------------------------------------------------------------
    # Lineage
    # ------------------------------------------------------------------

    def register_split(
        self,
        parent_path: str,
        splits: Dict[str, np.ndarray],
        split_params: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Register a train/cal/test split with lineage.

        Parameters
        ----------
        parent_path : str
            Path to the source dataset.
        splits : dict
            Mapping of split name → array (e.g., {'X_train': array, ...}).
        split_params : dict, optional
            Parameters used for splitting (ratios, random state, etc.).

        Returns
        -------
        dict
            Mapping of split name → version hash.
        """
        parent_hash = self._compute_checksum(Path(parent_path))

        version_map = {}
        for name, arr in splits.items():
            v = self.register_array(
                arr,
                name,
                description=f"Split '{name}' from {Path(parent_path).name}",
                parent_version=parent_hash,
            )
            if split_params:
                self._manifest[v]["metadata"]["split_params"] = split_params
            version_map[name] = v

        self._save_manifest()
        return version_map

    def get_lineage(self, version_hash: str) -> List[Dict]:
        """
        Trace the full lineage chain from a version back to its root.
        """
        chain = []
        current = version_hash

        while current and current in self._manifest:
            entry = self._manifest[current]
            chain.append(
                {
                    "version": current[:12],
                    "name": entry["file_name"],
                    "description": entry.get("description", ""),
                }
            )
            current = entry.get("parent_version")

        return chain

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self, file_path: str) -> bool:
        """Verify a file's integrity against the registered checksum."""
        file_path = Path(file_path)
        if not file_path.exists():
            return False

        current_hash = self._compute_checksum(file_path)
        if current_hash not in self._manifest:
            logger.warning(f"File {file_path.name} not found in registry")
            return False

        registered = self._manifest[current_hash]
        if registered["file_path"] != str(file_path.resolve()):
            logger.info(f"File path changed but content matches: {current_hash[:12]}")

        return True

    def verify_all(self) -> Dict[str, bool]:
        """Verify all registered file-based entries."""
        results = {}
        for checksum, entry in self._manifest.items():
            fp = entry["file_path"]
            if fp.startswith("<in-memory"):
                results[entry["file_name"]] = True  # Can't verify in-memory
                continue
            results[entry["file_name"]] = self.verify(fp)
        return results

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_version(self, version_hash: str) -> Optional[Dict]:
        return self._manifest.get(version_hash)

    def list_versions(self) -> List[Dict]:
        """List all registered data versions."""
        return [
            {"version": k[:12], **v}
            for k, v in sorted(
                self._manifest.items(),
                key=lambda x: x[1].get("registered_at", 0),
            )
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_checksum(file_path: Path) -> str:
        """Compute SHA-256 checksum of a file (streaming for large files)."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _load_manifest(self):
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                self._manifest = json.load(f)

    def _save_manifest(self):
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)
