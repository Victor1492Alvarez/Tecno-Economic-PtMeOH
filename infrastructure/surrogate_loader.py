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
    def __init__(self, bundle_path: Path):
        bundle = joblib.load(bundle_path)
        self.model_name = bundle["model_name"]
        self.input_column = bundle["input_column"]
        self.output_column = bundle["output_column"]
        self.scaler = bundle["scaler"]
        self.gpr = bundle["gpr"]

    def predict(self, h2_flow):
        arr = np.asarray(h2_flow, dtype=float).reshape(-1, 1)
        arr_scaled = self.scaler.transform(arr)
        pred, std = self.gpr.predict(arr_scaled, return_std=True)
        return pd.DataFrame(
            {
                self.input_column: arr.flatten(),
                "Prediction": pred,
                "Predictive Std": std,
                "Model Name": self.model_name,
                "Output Column": self.output_column,
            }
        )


class PythonWrapperPredictor:
    def __init__(self, module_path: Path, module_key: str):
        spec = importlib.util.spec_from_file_location(module_key, module_path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module

    def predict(self, h2_flow):
        result = self.module.predict(h2_flow)
        if isinstance(result, pd.DataFrame):
            return result
        return pd.DataFrame([result])


def load_surrogate_bundle(project_root: Path, model_name: str, library_name: str) -> SurrogateBundle:
    registry = ModelRegistry(project_root)
    files = registry.package_files(model_name, library_name)
    package_status = registry.inspect_package(model_name, library_name)
    parameters = registry.read_model_parameters(model_name, library_name)

    metadata = {}
    metadata_path = files["metadata"]
    if metadata_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    predictor = None
    joblib_path = files["joblib"]
    py_path = files["py"]

    if joblib_path is not None and joblib_path.exists():
        predictor = RuntimeJoblibPredictor(joblib_path)
    elif py_path is not None and py_path.exists():
        try:
            predictor = PythonWrapperPredictor(
                py_path,
                module_key=f"{library_name}__{model_name}",
            )
        except Exception:
            predictor = MockSurrogate(
                model_name=model_name,
                library_name=library_name,
                output_column=str(parameters.get("Output Column") or model_name),
            )
    else:
        predictor = MockSurrogate(
            model_name=model_name,
            library_name=library_name,
            output_column=str(parameters.get("Output Column") or model_name),
        )

    return SurrogateBundle(
        model_name=model_name,
        library_name=library_name,
        predictor=predictor,
        metadata=metadata,
        parameters=parameters,
        package_status=package_status,
    )
