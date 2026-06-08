from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ModelRegistry:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.catalog_path = self.project_root / "catalog.json"
        self.models_root = self.project_root / "models" / "packages"

    def _load_catalog_payload(self) -> dict:
        if self.catalog_path.exists():
            try:
                return json.loads(self.catalog_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _catalog_library_map(self) -> Dict[str, List[str]]:
        payload = self._load_catalog_payload()

        if isinstance(payload, dict):
            if "libraries" in payload and isinstance(payload["libraries"], dict):
                out: Dict[str, List[str]] = {}
                for lib, models in payload["libraries"].items():
                    if isinstance(models, list):
                        out[str(lib)] = [str(x) for x in models]
                    else:
                        out[str(lib)] = []
                return out

            out: Dict[str, List[str]] = {}
            for key, value in payload.items():
                if isinstance(value, list):
                    out[str(key)] = [str(x) for x in value]
            if out:
                return out

        return {}

    def _fs_library_map(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        if not self.models_root.exists():
            return out

        for library_dir in sorted([p for p in self.models_root.iterdir() if p.is_dir()]):
            model_names = sorted([p.name for p in library_dir.iterdir() if p.is_dir()])
            out[library_dir.name] = model_names
        return out

    def _merged_library_map(self) -> Dict[str, List[str]]:
        catalog_map = self._catalog_library_map()
        fs_map = self._fs_library_map()

        merged_keys = sorted(set(catalog_map.keys()) | set(fs_map.keys()))
        merged: Dict[str, List[str]] = {}

        for key in merged_keys:
            merged[key] = sorted(set(catalog_map.get(key, []) + fs_map.get(key, [])))
        return merged

    def get_library_names(self) -> List[str]:
        merged = self._merged_library_map()
        if merged:
            return sorted(merged.keys())
        return ["variable_h2_constant_co2", "variable_h2_variable_co2"]

    def get_models_by_library(self, library_name: str) -> List[str]:
        merged = self._merged_library_map()
        models = merged.get(str(library_name), [])
        return sorted(models)

    def _inspect_bundle(self, library_name: str, model_name: str) -> dict:
        bundle_dir = self.models_root / str(library_name) / str(model_name)

        joblib_files = sorted(bundle_dir.glob("*.joblib")) if bundle_dir.exists() else []
        py_files = sorted(bundle_dir.glob("*.py")) if bundle_dir.exists() else []
        txt_files = sorted(bundle_dir.glob("*.txt")) if bundle_dir.exists() else []

        joblib_found = len(joblib_files) > 0
        py_found = len(py_files) > 0
        txt_found = len(txt_files) > 0

        missing = []
        if not joblib_found:
            missing.append(".joblib")
        if not py_found:
            missing.append(".py")
        if not txt_found:
            missing.append(".txt")

        ready_for_runtime = joblib_found and py_found and txt_found
        ready_for_qa = py_found and txt_found

        return {
            "library": str(library_name),
            "model_name": str(model_name),
            "bundle_dir": str(bundle_dir),
            "joblib_found": bool(joblib_found),
            "py_found": bool(py_found),
            "txt_found": bool(txt_found),
            "ready_for_runtime": bool(ready_for_runtime),
            "ready_for_qa": bool(ready_for_qa),
            "missing_files": ", ".join(missing),
        }

    def discover_packages(self) -> pd.DataFrame:
        rows: list[dict] = []
        merged = self._merged_library_map()

        for library_name, model_names in merged.items():
            if not model_names:
                rows.append(
                    {
                        "library": str(library_name),
                        "model_name": "",
                        "bundle_dir": str(self.models_root / str(library_name)),
                        "joblib_found": False,
                        "py_found": False,
                        "txt_found": False,
                        "ready_for_runtime": False,
                        "ready_for_qa": False,
                        "missing_files": "no registered models",
                    }
                )
                continue

            for model_name in model_names:
                rows.append(self._inspect_bundle(library_name, model_name))

        if not rows:
            return pd.DataFrame(
                columns=[
                    "library",
                    "model_name",
                    "bundle_dir",
                    "joblib_found",
                    "py_found",
                    "txt_found",
                    "ready_for_runtime",
                    "ready_for_qa",
                    "missing_files",
                ]
            )

        return pd.DataFrame(rows).sort_values(["library", "model_name"]).reset_index(drop=True)

    # Optional compatibility aliases
    def catalog(self) -> pd.DataFrame:
        return self.discover_packages()

    def discover(self) -> pd.DataFrame:
        return self.discover_packages()

    def list_library_names(self) -> List[str]:
        return self.get_library_names()

    def list_models_by_library(self, library_name: str) -> List[str]:
        return self.get_models_by_library(library_name)
