from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from infrastructure.model_registry import ModelRegistry
from infrastructure.surrogate_loader import load_surrogate_bundle

class MultiSurrogateManager:
    DEFAULT_DOMAIN_MIN_KG_PER_H = 1.0
    DEFAULT_DOMAIN_MAX_KG_PER_H = 2.5

    def __init__(self, project_root: Path, library_name: str):
        self.project_root = Path(project_root)
        self.library_name = library_name
        self.registry = ModelRegistry(self.project_root)
        get_models = getattr(self.registry, 'get_models_by_library', None) or getattr(self.registry, 'getmodelsbylibrary')
        self.model_names = get_models(library_name)
        self.bundles = {name: load_surrogate_bundle(self.project_root, name, library_name) for name in self.model_names}
        self.domain_min, self.domain_max = self._infer_global_domain()

    def _infer_global_domain(self) -> tuple[float, float]:
        mins, maxs = [], []
        for bundle in self.bundles.values():
            if getattr(bundle, 'domain_min', None) is not None:
                mins.append(float(bundle.domain_min))
            if getattr(bundle, 'domain_max', None) is not None:
                maxs.append(float(bundle.domain_max))
        domain_min = max(mins) if mins else self.DEFAULT_DOMAIN_MIN_KG_PER_H
        domain_max = min(maxs) if maxs else self.DEFAULT_DOMAIN_MAX_KG_PER_H
        if domain_max < domain_min:
            domain_min, domain_max = min(domain_min, domain_max), max(domain_min, domain_max)
        return domain_min, domain_max

    def _predict_bundle_value(self, bundle, h2_flow_kg_per_h: float) -> tuple[float, float]:
        pred_df = bundle.predict([h2_flow_kg_per_h])
        pred_val = float(pred_df['Prediction'].iloc[0]) if 'Prediction' in pred_df.columns else float(pred_df.iloc[0, 1])
        std_val = float(pred_df['Predictive Std'].iloc[0]) if 'Predictive Std' in pred_df.columns else 0.0
        return pred_val, std_val

    def _edge_slope(self, bundle, edge: str, span_fraction: float = 0.01) -> float:
        bmin = float(bundle.domain_min if getattr(bundle, 'domain_min', None) is not None else self.domain_min)
        bmax = float(bundle.domain_max if getattr(bundle, 'domain_max', None) is not None else self.domain_max)
        span = max(bmax - bmin, 1e-6)
        delta = max(span * span_fraction, 1e-6)
        if edge == 'lower':
            x0, x1 = bmin, min(bmin + delta, bmax)
        else:
            x0, x1 = max(bmax - delta, bmin), bmax
        y0, _ = self._predict_bundle_value(bundle, x0)
        y1, _ = self._predict_bundle_value(bundle, x1)
        return (y1 - y0) / max(x1 - x0, 1e-9)

    def predict_all(self, h2_flow_kg_per_h: float, allow_soft_extrapolation_below_min: bool = True, soft_extrapolation_margin_fraction: float = 0.03) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        validity_checks: list[bool] = []
        records: list[dict[str, Any]] = []
        runtime_count = 0
        effective_domain_min = self.domain_min
        effective_domain_max = self.domain_max
        lower_soft_limit = effective_domain_min * (1.0 - max(soft_extrapolation_margin_fraction, 0.0))
        soft_extrapolated_any = False
        hard_out_any = False
        eval_mode = 'in_domain'

        for name, bundle in self.bundles.items():
            bundle_min = float(bundle.domain_min) if getattr(bundle, 'domain_min', None) is not None else effective_domain_min
            bundle_max = float(bundle.domain_max) if getattr(bundle, 'domain_max', None) is not None else effective_domain_max
            input_eval = float(h2_flow_kg_per_h)
            in_domain = bundle_min <= input_eval <= bundle_max
            soft_extrapolated = False
            hard_out_of_range = False

            if input_eval > bundle_max:
                input_eval = bundle_max
                in_domain = False
                hard_out_of_range = True
                hard_out_any = True
                eval_mode = 'hard_out_of_range'
            elif input_eval < bundle_min:
                if allow_soft_extrapolation_below_min and input_eval >= lower_soft_limit:
                    pred_edge, std_val = self._predict_bundle_value(bundle, bundle_min)
                    slope = self._edge_slope(bundle, 'lower')
                    pred_val = pred_edge + slope * (input_eval - bundle_min)
                    soft_extrapolated = True
                    soft_extrapolated_any = True
                    in_domain = False
                    if eval_mode != 'hard_out_of_range':
                        eval_mode = 'soft_extrapolated_below_min'
                else:
                    pred_edge, std_val = self._predict_bundle_value(bundle, bundle_min)
                    pred_val = pred_edge
                    hard_out_of_range = True
                    hard_out_any = True
                    in_domain = False
                    eval_mode = 'hard_out_of_range'
            else:
                pred_val, std_val = self._predict_bundle_value(bundle, input_eval)

            pred_val = max(float(pred_val), 0.0)
            validity_checks.append(in_domain)
            if getattr(bundle, 'runtime_mode', '') == 'runtime':
                runtime_count += 1
            output_name = bundle.output_column
            outputs[output_name] = pred_val
            outputs[f'{output_name}__std'] = std_val
            records.append({
                'library': self.library_name,
                'model_name': name,
                'output_name': output_name,
                'prediction': pred_val,
                'predictive_std': std_val,
                'input_column': getattr(bundle, 'input_column', 'H2'),
                'domain_min': bundle_min,
                'domain_max': bundle_max,
                'input_requested_kg_per_h': h2_flow_kg_per_h,
                'input_evaluated_kg_per_h': input_eval,
                'in_domain': in_domain,
                'soft_extrapolated': soft_extrapolated,
                'hard_out_of_range': hard_out_of_range,
                'runtime_mode': getattr(bundle, 'runtime_mode', 'unknown'),
                'ready_for_runtime': getattr(bundle, 'package_status', {}).get('ready_for_runtime', False),
                'missing_files': ', '.join(getattr(bundle, 'package_status', {}).get('missing_files', [])),
            })

        outputs['all_models_in_domain'] = all(validity_checks) if validity_checks else True
        outputs['soft_extrapolated_any'] = soft_extrapolated_any
        outputs['hard_out_of_range_any'] = hard_out_any
        outputs['eval_mode'] = eval_mode
        outputs['runtime_models_count'] = runtime_count
        outputs['total_models_count'] = len(self.bundles)
        outputs['effective_domain_min_kg_per_h'] = effective_domain_min
        outputs['effective_domain_max_kg_per_h'] = effective_domain_max
        outputs['model_summary_df'] = pd.DataFrame(records)
        return outputs
