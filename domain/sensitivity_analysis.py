from __future__ import annotations

from typing import Callable
import pandas as pd

from domain.data_models import (
    CaseInputs,
    ElectrolyzerInputs,
    PtMeOHInputs,
    ScenarioEconomicInputs,
    StorageInputs,
)
from domain.simulation_engine import SimulationEngine

ProgressCallback = Callable[[str, int, int], None] | None


class SensitivityAnalyzer:
    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.logger = getattr(engine, "logger", None)

    def _emit_progress(self, callback: ProgressCallback, stage: str, current: int, total: int) -> None:
        if callback is not None:
            callback(stage, current, total)

    def _clone_case(
        self,
        base_case: CaseInputs,
        *,
        electricity_price_multiplier: float = 1.0,
        electrolyzer_power_multiplier: float = 1.0,
        storage_multiplier: float = 1.0,
        target_h2_multiplier: float = 1.0,
    ) -> CaseInputs:
        new_power = max(base_case.electrolyzer.nominal_power_mw * electrolyzer_power_multiplier, 0.1)
        new_storage = max(base_case.storage.usable_capacity_kg_h2 * storage_multiplier, 0.0)
        new_target = max(base_case.ptmeoh.target_h2_feed_kg_per_h * target_h2_multiplier, 0.0)

        return CaseInputs(
            case_name=base_case.case_name,
            scenario_name=base_case.scenario_name,
            economic=ScenarioEconomicInputs(
                electricity_price_usd_per_mwh=base_case.economic.electricity_price_usd_per_mwh * electricity_price_multiplier,
                co2_price_usd_per_t=base_case.economic.co2_price_usd_per_t,
                methanol_price_usd_per_t=base_case.economic.methanol_price_usd_per_t,
                discount_rate=base_case.economic.discount_rate,
                project_years=base_case.economic.project_years,
                capex_multiplier=base_case.economic.capex_multiplier,
                opex_multiplier=base_case.economic.opex_multiplier,
            ),
            electrolyzer=ElectrolyzerInputs(
                nominal_power_mw=new_power,
                module_size_mw=max(new_power / max(1, len(base_case.optimization.module_count_grid)), 0.1),
                min_load_fraction=base_case.electrolyzer.min_load_fraction,
                specific_energy_kwh_per_kg_h2=base_case.electrolyzer.specific_energy_kwh_per_kg_h2,
                capex_usd_per_kw=base_case.electrolyzer.capex_usd_per_kw,
                fixed_opex_fraction=base_case.electrolyzer.fixed_opex_fraction,
            ),
            storage=StorageInputs(
                enabled=base_case.storage.enabled,
                usable_capacity_kg_h2=new_storage,
                initial_soc_fraction=base_case.storage.initial_soc_fraction,
                max_charge_kg_per_h=base_case.storage.max_charge_kg_per_h,
                max_discharge_kg_per_h=base_case.storage.max_discharge_kg_per_h,
                capex_usd_per_kg_h2=base_case.storage.capex_usd_per_kg_h2,
            ),
            ptmeoh=PtMeOHInputs(
                operating_mode=base_case.ptmeoh.operating_mode,
                surrogate_library=base_case.ptmeoh.surrogate_library,
                target_h2_feed_kg_per_h=min(new_target, base_case.ptmeoh.max_h2_feed_kg_per_h),
                max_h2_feed_kg_per_h=base_case.ptmeoh.max_h2_feed_kg_per_h,
                methanol_yield_t_meoh_per_t_h2=base_case.ptmeoh.methanol_yield_t_meoh_per_t_h2,
                unmet_h2_warning_threshold=base_case.ptmeoh.unmet_h2_warning_threshold,
                curtailment_warning_threshold=base_case.ptmeoh.curtailment_warning_threshold,
                utilization_warning_threshold=base_case.ptmeoh.utilization_warning_threshold,
            ),
            optimization=base_case.optimization,
            renewable_profile=base_case.renewable_profile.copy(),
            time_step_h=base_case.time_step_h,
        )

    def run(self, base_case: CaseInputs, progress_callback: ProgressCallback = None) -> pd.DataFrame:
        experiments = [
            ("electricity_price", 0.85, 1.15),
            ("electrolyzer_power", 0.90, 1.10),
            ("storage_capacity", 0.80, 1.20),
            ("target_h2_feed", 0.90, 1.10),
        ]

        total = len(experiments) * 2
        current = 0
        rows: list[dict] = []
        self._emit_progress(progress_callback, "sensitivity", 0, max(total, 1))

        for parameter, low_mult, high_mult in experiments:
            for label, mult in [("low", low_mult), ("high", high_mult)]:
                if parameter == "electricity_price":
                    case = self._clone_case(base_case, electricity_price_multiplier=mult)
                elif parameter == "electrolyzer_power":
                    case = self._clone_case(base_case, electrolyzer_power_multiplier=mult)
                elif parameter == "storage_capacity":
                    case = self._clone_case(base_case, storage_multiplier=mult)
                else:
                    case = self._clone_case(base_case, target_h2_multiplier=mult)

                sim = self.engine.run(case)

                row = {
                    "parameter": parameter,
                    "case_label": label,
                    "multiplier": mult,
                    **sim.kpis,
                    "warning_count": len(sim.warnings),
                }
                rows.append(row)

                current += 1
                self._emit_progress(progress_callback, "sensitivity", current, max(total, 1))

        return pd.DataFrame(rows)
