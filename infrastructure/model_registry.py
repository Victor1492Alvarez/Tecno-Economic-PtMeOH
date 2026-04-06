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
            return {"models": {}}
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def package_path(self, model_name: str) -> Path:
        return self.package_root / model_name

    def discover_model_names(self) -> list[str]:
        if not self.package_root.exists():
            return []
        return sorted([p.name for p in self.package_root.iterdir() if p.is_dir()])

    def get_models_by_library(self, library_name: str) -> list[str]:
        catalog = self.load_catalog().get("models", {})
        configured = [name for name, meta in catalog.items() if meta.get("library") == library_name]
        discovered = self.discover_model_names()
        present = [m for m in configured if m in discovered]
        return present or configured

    def package_files(self, model_name: str) -> dict[str, Path]:
        pkg = self.package_path(model_name)
        return {
            "joblib": pkg / f"{model_name}.joblib",
            "py": pkg / f"{model_name}.py",
            "txt": pkg / f"{model_name}.txt",
            "metadata": pkg / "metadata.json",
            "parameters": pkg / "model_parameters.xlsx",
            "consolidated_report": pkg / "consolidated_model_report.pdf",
            "training_report": pkg / "training_validation_report.pdf",
        }

    def inspect_package(self, model_name: str) -> dict[str, Any]:
        files = self.package_files(model_name)
        file_status = {k: v.exists() for k, v in files.items()}
        return {
            "model_name": model_name,
            "folder_exists": self.package_path(model_name).exists(),
            "file_status": file_status,
            "ready_for_runtime": file_status["joblib"] and file_status["py"],
            "ready_for_qa": all(file_status[k] for k in ["metadata", "parameters", "consolidated_report", "training_report"]),
            "missing_files": [name for name, ok in file_status.items() if not ok],
        }

    def discover_packages(self) -> pd.DataFrame:
        catalog = self.load_catalog().get("models", {})
        names = sorted(set(list(catalog.keys()) + self.discover_model_names()))
        rows = []
        for model_name in names:
            inspected = self.inspect_package(model_name)
            meta = catalog.get(model_name, {})
            rows.append({
                "model_name": model_name,
                "library": meta.get("library", "unclassified"),
                "category": meta.get("category", "unknown"),
                "unit": meta.get("unit", "unknown"),
                "folder_exists": inspected["folder_exists"],
                "ready_for_runtime": inspected["ready_for_runtime"],
                "ready_for_qa": inspected["ready_for_qa"],
                "missing_files": ", ".join(inspected["missing_files"]),
            })
        return pd.DataFrame(rows)

    def read_model_parameters(self, model_name: str) -> dict:
        path = self.package_files(model_name)["parameters"]
        if not path.exists():
            return {}
        xl = pd.ExcelFile(path)
        meta_sheet = "Model Metadata" if "Model Metadata" in xl.sheet_names else xl.sheet_names
        log_sheet = "Temporary Parameter Log" if "Temporary Parameter Log" in xl.sheet_names else None
        meta = pd.read_excel(path, sheet_name=meta_sheet)
        row = meta.iloc.to_dict() if not meta.empty else {}
        if log_sheet:
            fold = pd.read_excel(path, sheet_name=log_sheet)
            if not fold.empty:
                if "Train X Min" in fold.columns:
                    row["train_x_min"] = float(fold["Train X Min"].min())
                if "Train X Max" in fold.columns:
                    row["train_x_max"] = float(fold["Train X Max"].max())
        return row

