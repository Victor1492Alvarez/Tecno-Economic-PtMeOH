from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from infrastructure.model_registry import ModelRegistry
from infrastructure.surrogate_loader import load_surrogate_bundle

class MultiSurrogateManager:
    def __init__(self, project_root: Path, library_name: str):
        self.project_root = Path(project_root)
        self.library_name = library_name
        self.registry = ModelRegistry(self.project_root)
        self.model_names = self.registry.get_models_by_library(library_name)
        self.bundles = {name: load_surrogate_bundle(self.project_root, name) for name in self.model_names}

    def predict_all(self, h2_flow_kg_per_h: float) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        validity_checks: list[bool] = []
        records: list[dict[str, Any]] = []
        runtime_count = 0
        for name, bundle in self.bundles.items():
            pred_df = bundle.predict([h2_flow_kg_per_h])
            pred_val = float(pred_df["Prediction"].iloc) if "Prediction" in pred_df.columns else float(pred_df.iloc[0, 1])
            std_val = float(pred_df["Predictive Std"].iloc) if "Predictive Std" in pred_df.columns else 0.0
            in_domain = True
            if bundle.domain_min is not None and h2_flow_kg_per_h < float(bundle.domain_min):
                in_domain = False
            if bundle.domain_max is not None and h2_flow_kg_per_h > float(bundle.domain_max):
                in_domain = False
            validity_checks.append(in_domain)
            if bundle.runtime_mode == "runtime":
                runtime_count += 1
            output_name = bundle.output_column
            outputs[output_name] = pred_val
            outputs[f"{output_name}__std"] = std_val
            records.append({
                "model_name": name,
                "output_name": output_name,
                "prediction": pred_val,
                "predictive_std": std_val,
                "input_column": bundle.input_column,
                "domain_min": bundle.domain_min,
                "domain_max": bundle.domain_max,
                "in_domain": in_domain,
                "runtime_mode": bundle.runtime_mode,
                "ready_for_runtime": bundle.package_status.get("ready_for_runtime", False),
                "missing_files": ", ".join(bundle.package_status.get("missing_files", [])),
            })
        outputs["all_models_in_domain"] = all(validity_checks) if validity_checks else True
        outputs["runtime_models_count"] = runtime_count
        outputs["total_models_count"] = len(self.bundles)
        outputs["model_summary_df"] = pd.DataFrame(records)
        return outputs


