from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

from domain.data_models import (
    CaseInputs,
    ElectrolyzerInputs,
    OptimizationInputs,
    PtMeOHInputs,
    ScenarioEconomicInputs,
    StorageInputs,
)
from domain.renewable_profile import build_default_hourly_profile
from domain.simulation_engine import SimulationEngine
from domain.optimizer_grid import GridOptimizer
from domain.sensitivity_analysis import SensitivityAnalyzer

ProgressCallback = Callable[[str, int, int], None] | None


class CaseRunner:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.engine = SimulationEngine(self.project_root)
        self.optimizer = GridOptimizer(self.engine)
        self.sensitivity = SensitivityAnalyzer(self.engine)
        self.logger = getattr(self.engine, "logger", None)
        self.scenario_config = json.loads(
            (self.project_root / "config" / "scenarios.json").read_text(encoding="utf-8")
        )

    def build_case(
        self,
        scenario_name: str,
        electrolyzer_power_mw: float,
        module_count: int,
        storage_enabled: bool,
        storage_kg_h2: float,
        operating_mode: str,
        surrogate_library: str,
        target_h2_kg_per_h: float,
        max_h2_feed_kg_per_h: float,
        renewable_peak_power_mw: float,
        renewable_profile_df: pd.DataFrame | None = None,
        electricity_price_usd_per_kwh: float | None = None,
    ) -> CaseInputs:
        econ_payload = dict(self.scenario_config[scenario_name])

        if electricity_price_usd_per_kwh is not None:
            econ_payload["electricity_price_usd_per_mwh"] = float(electricity_price_usd_per_kwh) * 1000.0

        econ = ScenarioEconomicInputs(**econ_payload)

        if renewable_profile_df is None:
            renewable_profile = build_default_hourly_profile(peak_power_mw=float(renewable_peak_power_mw))
        else:
            renewable_profile = renewable_profile_df.copy()
            if "timestamp" in renewable_profile.columns:
                renewable_profile["timestamp"] = pd.to_datetime(
                    renewable_profile["timestamp"], errors="coerce"
                )
            if "renewable_power_mw" in renewable_profile.columns:
                renewable_profile["renewable_power_mw"] = pd.to_numeric(
                    renewable_profile["renewable_power_mw"], errors="coerce"
                )
            renewable_profile = renewable_profile.dropna().reset_index(drop=True)

        electrolyzer_power_mw = float(electrolyzer_power_mw)
        module_count = int(module_count)
        storage_kg_h2 = float(storage_kg_h2)
        target_h2_kg_per_h = float(target_h2_kg_per_h)
        max_h2_feed_kg_per_h = float(max_h2_feed_kg_per_h)

        electrolyzer = ElectrolyzerInputs(
            nominal_power_mw=electrolyzer_power_mw,
            module_size_mw=max(electrolyzer_power_mw / max(module_count, 1), 0.1),
            min_load_fraction=0.15,
            specific_energy_kwh_per_kg_h2=52.0,
            capex_usd_per_kw=850.0,
            fixed_opex_fraction=0.03,
        )

        storage = StorageInputs(
            enabled=bool(storage_enabled),
            usable_capacity_kg_h2=storage_kg_h2,
            initial_soc_fraction=0.30,
        )

        ptmeoh = PtMeOHInputs(
            operating_mode=str(operating_mode),
            surrogate_library=str(surrogate_library),
            target_h2_feed_kg_per_h=target_h2_kg_per_h,
            max_h2_feed_kg_per_h=max_h2_feed_kg_per_h,
        )

        storage_grid = {
            0.0 if storage_kg_h2 <= 0 else round(0.5 * storage_kg_h2, 2),
            round(storage_kg_h2, 2),
            500.0 if storage_kg_h2 <= 0 else round(1.5 * storage_kg_h2, 2),
        }

        optimization = OptimizationInputs(
            electrolyzer_power_grid_mw=sorted(
                {
                    round(0.8 * electrolyzer_power_mw, 2),
                    round(electrolyzer_power_mw, 2),
                    round(1.2 * electrolyzer_power_mw, 2),
                }
            ),
            storage_grid_kg_h2=sorted(storage_grid),
            target_h2_grid_kg_per_h=sorted(
                {
                    round(0.85 * target_h2_kg_per_h, 2),
                    round(target_h2_kg_per_h, 2),
                    round(min(1.15 * target_h2_kg_per_h, max_h2_feed_kg_per_h), 2),
                }
            ),
            module_count_grid=sorted({max(module_count - 1, 1), module_count, module_count + 1}),
        )

        return CaseInputs(
            case_name="base_case",
            scenario_name=str(scenario_name),
            economic=econ,
            electrolyzer=electrolyzer,
            storage=storage,
            ptmeoh=ptmeoh,
            optimization=optimization,
            renewable_profile=renewable_profile,
            time_step_h=1.0,
        )

    def run_simulation(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
    ):
        if self.logger is not None:
            self.logger.info("Running base simulation")
        return self.engine.run(case, progress_callback=progress_callback)

    def run_optimization(
        self,
        case: CaseInputs,
        progress_callback: ProgressCallback = None,
    ):
        if self.logger is not None:
            self.logger.info("Running optimization")
        return self.optimizer.run(case, progress_callback=progress_callback)

    def run_sensitivity(self, case: CaseInputs):
        if self.logger is not None:
            self.logger.info("Running sensitivity analysis")
        return self.sensitivity.run(case)

    def run_all(
        self,
        case: CaseInputs,
        run_optimization: bool = True,
        run_sensitivity: bool = True,
        progress_callback: ProgressCallback = None,
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}
        outputs["simulation"] = self.run_simulation(case, progress_callback=progress_callback)

        if run_optimization:
            outputs["optimization"] = self.run_optimization(case, progress_callback=progress_callback)
        else:
            if self.logger is not None:
                self.logger.info("Optimization skipped by user")
            outputs["optimization"] = None

        if run_sensitivity:
            sensitivity_result = self.run_sensitivity(case)
            outputs["sensitivity"] = sensitivity_result
            outputs["sensitivities"] = sensitivity_result
        else:
            if self.logger is not None:
                self.logger.info("Sensitivity analysis skipped by user")
            outputs["sensitivity"] = None
            outputs["sensitivities"] = None

        return outputs
