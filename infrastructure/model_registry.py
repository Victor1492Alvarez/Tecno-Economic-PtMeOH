from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd


class ModelRegistry:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.catalog_path = self.project_root / "models" / "catalog" / "catalog.json"
        self.package_root = self.project_root / "models" / "packages"

    def load_catalog(self) -> dict[str, Any]:
        if not self.catalog_path.exists():
            return {"libraries": [], "model_order": [], "models": {}}
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _meta_libraries(self, meta: dict[str, Any]) -> list[str]:
        if "libraries" in meta and isinstance(meta["libraries"], list):
            return [str(x) for x in meta["libraries"]]
        if "library" in meta and meta["library"]:
            return [str(meta["library"])]
        return []

    def get_library_names(self) -> list[str]:
        catalog = self.load_catalog()
        configured = [str(x) for x in catalog.get("libraries", [])]

        discovered = []
        if self.package_root.exists():
            discovered = sorted([p.name for p in self.package_root.iterdir() if p.is_dir()])

        names: list[str] = []
        for item in configured + discovered:
            if item not in names:
                names.append(item)
        return names

    def get_model_order(self) -> list[str]:
        catalog = self.load_catalog()
        order = [str(x) for x in catalog.get("model_order", [])]
        if order:
            return order
        return list(catalog.get("models", {}).keys())

    def package_path(self, model_name: str, library_name: str) -> Path:
        return self.package_root / library_name / model_name

    def discover_model_names(self, library_name: str) -> list[str]:
        lib_root = self.package_root / library_name
        if not lib_root.exists():
            return []
        return sorted([p.name for p in lib_root.iterdir() if p.is_dir()])

    def get_models_by_library(self, library_name: str) -> list[str]:
        catalog = self.load_catalog().get("models", {})
        ordered = [
            model_name
            for model_name in self.get_model_order()
            if library_name in self._meta_libraries(catalog.get(model_name, {}))
        ]
        discovered = self.discover_model_names(library_name)
        extras = [m for m in discovered if m not in ordered]
        return ordered + extras

    def _first_match(self, pkg: Path, patterns: list[str]) -> Path | None:
        for pattern in patterns:
            matches = sorted(pkg.glob(pattern))
            if matches:
                return matches[0]
        return None

    def package_files(self, model_name: str, library_name: str) -> dict[str, Path | None]:
        pkg = self.package_path(model_name, library_name)

        if not pkg.exists():
            return {
                "joblib": None,
                "py": None,
                "txt": None,
                "metadata": None,
                "parameters": None,
                "consolidated_report": None,
                "training_report": None,
            }

        return {
            "joblib": self._first_match(pkg, [f"{model_name}.joblib", "*.joblib"]),
            "py": self._first_match(pkg, [f"{model_name}.py", "*.py"]),
            "txt": self._first_match(pkg, [f"{model_name}.txt", "*.txt"]),
            "metadata": self._first_match(pkg, ["metadata.json", "*.json"]),
            "parameters": self._first_match(pkg, ["model_parameters.xlsx", "*.xlsx", "*.xls"]),
            "consolidated_report": self._first_match(
                pkg,
                ["consolidated_model_report.pdf", "*consolidated*.pdf", "*.pdf"],
            ),
            "training_report": self._first_match(
                pkg,
                ["training_validation_report.pdf", "*training*.pdf", "*.pdf"],
            ),
        }

    def inspect_package(self, model_name: str, library_name: str) -> dict[str, Any]:
        pkg = self.package_path(model_name, library_name)
        files = self.package_files(model_name, library_name)
        file_status = {k: (v is not None and v.exists()) for k, v in files.items()}
        file_names = {k: (v.name if v is not None else "") for k, v in files.items()}

        return {
            "model_name": model_name,
            "library": library_name,
            "folder_exists": pkg.exists(),
            "file_status": file_status,
            "file_names": file_names,
            "ready_for_runtime": file_status["joblib"] and file_status["py"],
            "ready_for_qa": all(
                file_status[k]
                for k in ["metadata", "parameters", "consolidated_report", "training_report"]
            ),
            "missing_files": [name for name, ok in file_status.items() if not ok],
        }

    def discover_packages(self) -> pd.DataFrame:
        catalog = self.load_catalog().get("models", {})
        libraries = self.get_library_names()
        rows = []

        for library_name in libraries:
            names = self.get_models_by_library(library_name)
            for model_name in names:
                inspected = self.inspect_package(model_name, library_name)
                meta = catalog.get(model_name, {})
                rows.append(
                    {
                        "library": library_name,
                        "model_name": model_name,
                        "category": meta.get("category", "unknown"),
                        "unit": meta.get("unit", "unknown"),
                        "folder_exists": inspected["folder_exists"],
                        "joblib_found": inspected["file_status"]["joblib"],
                        "py_found": inspected["file_status"]["py"],
                        "txt_found": inspected["file_status"]["txt"],
                        "joblib_file": inspected["file_names"]["joblib"],
                        "py_file": inspected["file_names"]["py"],
                        "txt_file": inspected["file_names"]["txt"],
                        "ready_for_runtime": inspected["ready_for_runtime"],
                        "ready_for_qa": inspected["ready_for_qa"],
                        "missing_files": ", ".join(inspected["missing_files"]),
                    }
                )

        return pd.DataFrame(rows)

    def read_model_parameters(self, model_name: str, library_name: str) -> dict:
        path = self.package_files(model_name, library_name)["parameters"]
        if path is None or not path.exists():
            return {}

        xl = pd.ExcelFile(path)
        meta_sheet = "Model Metadata" if "Model Metadata" in xl.sheet_names else xl.sheet_names[0]
        log_sheet = "Temporary Parameter Log" if "Temporary Parameter Log" in xl.sheet_names else None

        meta = pd.read_excel(path, sheet_name=meta_sheet)
        row = meta.iloc[0].to_dict() if not meta.empty else {}

        if log_sheet:
            fold = pd.read_excel(path, sheet_name=log_sheet)
            if not fold.empty:
                if "Train X Min" in fold.columns:
                    row["train_x_min"] = float(fold["Train X Min"].min())
                if "Train X Max" in fold.columns:
                    row["train_x_max"] = float(fold["Train X Max"].max())

        return row
