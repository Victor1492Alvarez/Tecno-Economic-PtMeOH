from __future__ import annotations
from pathlib import Path
import json
from domain.data_models import CaseInputs, ElectrolyzerInputs, OptimizationInputs, PtMeOHInputs, ScenarioEconomicInputs, StorageInputs
from domain.renewable_profile import build_default_hourly_profile
from domain.simulation_engine import SimulationEngine
from domain.optimizer_grid import GridOptimizer
from domain.sensitivity_analysis import SensitivityAnalyzer

class CaseRunner:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.engine = SimulationEngine(self.project_root)
        self.optimizer = GridOptimizer(self.engine)
        self.sensitivity = SensitivityAnalyzer(self.engine)
        self.scenario_config = json.loads((self.project_root / "config" / "scenarios.json").read_text(encoding="utf-8"))

    def build_case(self, scenario_name: str, electrolyzer_power_mw: float, module_count: int, storage_enabled: bool, storage_kg_h2: float, operating_mode: str, surrogate_library: str, target_h2_kg_per_h: float, max_h2_feed_kg_per_h: float, renewable_peak_power_mw: float) -> CaseInputs:
        econ = ScenarioEconomicInputs(**self.scenario_config[scenario_name])
        renewable_profile = build_default_hourly_profile(peak_power_mw=renewable_peak_power_mw)
        electrolyzer = ElectrolyzerInputs(
            nominal_power_mw=electrolyzer_power_mw,
            module_size_mw=max(electrolyzer_power_mw / max(module_count, 1), 0.1),
            min_load_fraction=0.15,
            specific_energy_kwh_per_kg_h2=52.0,
            capex_usd_per_kw=850.0,
            fixed_opex_fraction=0.03,
        )
        storage = StorageInputs(enabled=storage_enabled, usable_capacity_kg_h2=storage_kg_h2, initial_soc_fraction=0.30)
        ptmeoh = PtMeOHInputs(
            operating_mode=operating_mode,
            surrogate_library=surrogate_library,
            target_h2_feed_kg_per_h=target_h2_kg_per_h,
            max_h2_feed_kg_per_h=max_h2_feed_kg_per_h,
        )
        optimization = OptimizationInputs(
            electrolyzer_power_grid_mw=sorted({round(0.8 * electrolyzer_power_mw, 2), round(electrolyzer_power_mw, 2), round(1.2 * electrolyzer_power_mw, 2)}),
            storage_grid_kg_h2=sorted({round(0.5 * storage_kg_h2, 2) if storage_kg_h2 > 0 else 0.0, round(storage_kg_h2, 2), round(1.5 * storage_kg_h2, 2) if storage_kg_h2 > 0 else 500.0}),
            target_h2_grid_kg_per_h=sorted({round(0.85 * target_h2_kg_per_h, 2), round(target_h2_kg_per_h, 2), round(min(1.15 * target_h2_kg_per_h, max_h2_feed_kg_per_h), 2)}),
            module_count_grid=sorted({max(module_count - 1, 1), module_count, module_count + 1}),
        )
        return CaseInputs(
            case_name="base_case",
            scenario_name=scenario_name,
            economic=econ,
            electrolyzer=electrolyzer,
            storage=storage,
            ptmeoh=ptmeoh,
            optimization=optimization,
            renewable_profile=renewable_profile,
            time_step_h=1.0,
        )

    def run_all(self, case: CaseInputs) -> dict:
        simulation = self.engine.run(case)
        optimization = self.optimizer.run(case)
        sensitivities = self.sensitivity.run(case)
        return {
            "simulation": simulation,
            "optimization": optimization,
            "sensitivities": sensitivities,
        }
