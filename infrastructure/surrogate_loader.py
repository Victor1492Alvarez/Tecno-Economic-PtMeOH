from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd

from infrastructure.model_registry import ModelRegistry


class MockSurrogate:
    def __init__(
        self,
        model_name: str,
        library_name: str,
        input_column: str = "h2_flow_kg_per_h",
        output_column: str = "prediction",
    ):
        self.model_name = model_name
        self.library_name = library_name
        self.input_column = input_column
        self.output_column = output_column

    def predict(self, h2_flow):
        arr = np.asarray(h2_flow, dtype=float).reshape(-1)
        seed = (sum(ord(c) for c in f"{self.library_name}:{self.model_name}") % 17) + 1
        slope = 0.08 + 0.01 * seed
        pred = slope * arr + 0.01 * np.sin(arr / max(seed, 1))
        std = np.full_like(arr, 0.03, dtype=float)
        return pd.DataFrame(
            {
                self.input_column: arr,
                "Prediction": pred,
                "Predictive Std": std,
                "Model Name": self.model_name,
                "Output Column": self.output_column,
            }
        )


class SurrogateBundle:
    def __init__(
        self,
        model_name: str,
        library_name: str,
        predictor,
        metadata: dict,
        parameters: dict,
        package_status: dict,
    ):
        self.model_name = model_name
        self.library_name = library_name
        self.predictor = predictor
        self.metadata = metadata
        self.parameters = parameters
        self.package_status = package_status

    @property
    def input_column(self) -> str:
        return str(
            self.parameters.get("Input Column")
            or self.metadata.get("input_column")
            or "h2_flow"
        )

    @property
    def output_column(self) -> str:
        return str(
            self.parameters.get("Output Column")
            or self.metadata.get("output_column")
            or self.model_name
        )

    @property
    def domain_min(self):
        return self.parameters.get("train_x_min")

    @property
    def domain_max(self):
        return self.parameters.get("train_x_max")

    @property
    def runtime_mode(self) -> str:
        return "runtime" if self.package_status.get("ready_for_runtime") else "mock"

    def predict(self, h2_flow):
        return self.predictor.predict(h2_flow)


class RuntimeJoblibPredictor:
    def __init__(self, model_object, model_name: str, input_column: str, output_column: str):
        self.model_object = model_object
        self.model_name = model_name
        self.input_column = input_column
        self.output_column = output_column

    def _predict_raw(self, arr: np.ndarray):
        model = self.model_object
        x_df = pd.DataFrame({self.input_column: arr})
        x_np = arr.reshape(-1, 1)

        last_exc = None
        for payload in (x_df, x_np, arr):
            try:
                return model.predict(payload)
            except Exception as exc:
                last_exc = exc

        raise RuntimeError(f"Model '{self.model_name}' prediction failed: {last_exc}")

    def predict(self, h2_flow):
        arr = np.asarray(h2_flow, dtype=float).reshape(-1)
        raw = self._predict_raw(arr)
        std = np.zeros_like(arr, dtype=float)

        if isinstance(raw, tuple) and len(raw) >= 2:
            pred, maybe_std = raw[0], raw[1]
            pred = np.asarray(pred, dtype=float).reshape(-1)
            try:
                std = np.asarray(maybe_std, dtype=float).reshape(-1)
            except Exception:
                std = np.zeros_like(pred, dtype=float)
        elif isinstance(raw, pd.DataFrame):
            if "Prediction" in raw.columns:
                pred = raw["Prediction"].to_numpy(dtype=float)
            else:
                pred = raw.iloc[:, -1].to_numpy(dtype=float)

            if "Predictive Std" in raw.columns:
                std = raw["Predictive Std"].to_numpy(dtype=float)
            else:
                std = np.zeros_like(pred, dtype=float)
        else:
            pred = np.asarray(raw, dtype=float).reshape(-1)

        if len(std) != len(pred):
            std = np.zeros_like(pred, dtype=float)

        return pd.DataFrame(
            {
                self.input_column: arr,
                "Prediction": pred,
                "Predictive Std": std,
                "Model Name": self.model_name,
                "Output Column": self.output_column,
            }
        )


def _safe_parse_scalar(text: str):
    s = text.strip()
    if s == "":
        return s
    lowered = s.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in s or "e" in lowered:
            return float(s)
        return int(s)
    except Exception:
        return s


def _load_txt_parameters(txt_path: Path) -> dict:
    if not txt_path.exists():
        return {}

    text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    params: dict = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
        elif "=" in stripped:
            key, value = stripped.split("=", 1)
        else:
            continue
        params[key.strip()] = _safe_parse_scalar(value)
    return params


def _load_py_metadata(py_path: Path) -> dict:
    if not py_path.exists():
        return {}

    try:
        spec = importlib.util.spec_from_file_location(py_path.stem, py_path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        meta = {}
        for key in [
            "INPUT_COLUMN",
            "OUTPUT_COLUMN",
            "input_column",
            "output_column",
            "TRAIN_X_MIN",
            "TRAIN_X_MAX",
            "train_x_min",
            "train_x_max",
        ]:
            if hasattr(module, key):
                meta[key.lower()] = getattr(module, key)
        return meta
    except Exception:
        return {}


def load_surrogate_bundle(project_root: Path, model_name: str, library_name: str) -> SurrogateBundle:
    project_root = Path(project_root)
    bundle_dir = project_root / "models" / "packages" / library_name / model_name

    registry = ModelRegistry(project_root)
    catalog = registry.discover_packages()
    row = None
    if not catalog.empty and {"library", "model_name"}.issubset(catalog.columns):
        matched = catalog[
            (catalog["library"].astype(str) == str(library_name))
            & (catalog["model_name"].astype(str) == str(model_name))
        ]
        if not matched.empty:
            row = matched.iloc[0].to_dict()

    joblib_files = sorted(bundle_dir.glob("*.joblib")) if bundle_dir.exists() else []
    py_files = sorted(bundle_dir.glob("*.py")) if bundle_dir.exists() else []
    txt_files = sorted(bundle_dir.glob("*.txt")) if bundle_dir.exists() else []

    ready_for_runtime = bool(row.get("ready_for_runtime")) if isinstance(row, dict) else (
        bool(joblib_files) and bool(py_files) and bool(txt_files)
    )

    missing_files = []
    if isinstance(row, dict) and "missing_files" in row:
        if isinstance(row["missing_files"], str) and row["missing_files"].strip():
            missing_files = [x.strip() for x in row["missing_files"].split(",") if x.strip()]
    else:
        if not joblib_files:
            missing_files.append(".joblib")
        if not py_files:
            missing_files.append(".py")
        if not txt_files:
            missing_files.append(".txt")

    metadata = _load_py_metadata(py_files[0]) if py_files else {}
    parameters = _load_txt_parameters(txt_files[0]) if txt_files else {}

    input_column = str(parameters.get("Input Column") or metadata.get("input_column") or "h2_flow_kg_per_h")
    output_column = str(parameters.get("Output Column") or metadata.get("output_column") or model_name)

    package_status = {
        "ready_for_runtime": ready_for_runtime,
        "missing_files": missing_files,
    }

    if ready_for_runtime and joblib_files:
        try:
            model_object = joblib.load(joblib_files[0])
            predictor = RuntimeJoblibPredictor(
                model_object=model_object,
                model_name=model_name,
                input_column=input_column,
                output_column=output_column,
            )
        except Exception:
            predictor = MockSurrogate(
                model_name=model_name,
                library_name=library_name,
                input_column=input_column,
                output_column=output_column,
            )
            package_status["ready_for_runtime"] = False
            if ".joblib_load_failed" not in package_status["missing_files"]:
                package_status["missing_files"].append(".joblib_load_failed")
    else:
        predictor = MockSurrogate(
            model_name=model_name,
            library_name=library_name,
            input_column=input_column,
            output_column=output_column,
        )

    return SurrogateBundle(
        model_name=model_name,
        library_name=library_name,
        predictor=predictor,
        metadata=metadata,
        parameters=parameters,
        package_status=package_status,
    )
