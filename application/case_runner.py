from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
from domain.data_models import CaseInputs, ElectrolyzerInputs, OptimizationInputs, PtMeOHInputs, ScenarioEconomicInputs, StorageInputs
from domain.optimizer_grid import GridOptimizer
from domain.simulation_engine import SimulationEngine
from domain.ptmeoh_surrogate import MultiSurrogateManager

class CaseRunner:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger('ptmeoh_tool')
        self.engine = SimulationEngine(self.project_root)
        self.optimizer = GridOptimizer(self.engine)
        self.scenario_config = {
            'optimistic': {'electricity_price_usd_per_mwh': 35.0, 'co2_price_usd_per_t': 30.0, 'methanol_price_usd_per_t': 450.0, 'discount_rate': 0.08, 'project_years': 20},
            'moderate': {'electricity_price_usd_per_mwh': 55.0, 'co2_price_usd_per_t': 60.0, 'methanol_price_usd_per_t': 400.0, 'discount_rate': 0.10, 'project_years': 20},
            'pessimistic': {'electricity_price_usd_per_mwh': 80.0, 'co2_price_usd_per_t': 90.0, 'methanol_price_usd_per_t': 350.0, 'discount_rate': 0.12, 'project_years': 20},
        }

    def _synthetic_profile(self, renewable_peak_power_mw: float) -> pd.DataFrame:
        return self.engine._default_profile(renewable_peak_power_mw)

    def build_case(
        self,
        scenario_name: str,
        electrolyzer_power_mw: float,
        module_count: int,
        storage_enabled: bool,
        storage_kg_h2: float,
        operating_mode: str,
        surrogate_library: str,
        renewable_peak_power_mw: float,
        renewable_profile_df: pd.DataFrame | None = None,
        electricity_price_usd_per_kwh: float | None = None,
        allow_soft_extrapolation_below_min: bool = True,
        soft_extrapolation_margin_fraction: float = 0.03,
    ) -> CaseInputs:
        scenario = dict(self.scenario_config.get(scenario_name, self.scenario_config['moderate']))
        if electricity_price_usd_per_kwh is not None:
            scenario['electricity_price_usd_per_mwh'] = float(electricity_price_usd_per_kwh) * 1000.0

        manager = MultiSurrogateManager(self.project_root, surrogate_library)
        domain_min = float(manager.domain_min)
        domain_max = float(manager.domain_max)
        renewable_profile = renewable_profile_df.copy() if renewable_profile_df is not None else self._synthetic_profile(renewable_peak_power_mw)

        return CaseInputs(
            scenario_name=scenario_name,
            case_name='base_case',
            economic=ScenarioEconomicInputs(**scenario),
            electrolyzer=ElectrolyzerInputs(
                nominal_power_mw=float(electrolyzer_power_mw),
                module_size_mw=max(float(electrolyzer_power_mw) / max(int(module_count), 1), 0.1),
                min_load_fraction=0.20,
                specific_energy_kwh_per_kg_h2=50.0,
                capex_usd_per_kw=800.0,
                fixed_opex_fraction=0.03,
            ),
            storage=StorageInputs(
                enabled=bool(storage_enabled),
                usable_capacity_kg_h2=float(storage_kg_h2),
                initial_soc_fraction=0.50,
                max_charge_kg_per_h=float(storage_kg_h2) if storage_enabled else 0.0,
                max_discharge_kg_per_h=float(storage_kg_h2) if storage_enabled else 0.0,
                capex_usd_per_kg_h2=20.0,
            ),
            ptmeoh=PtMeOHInputs(
                operating_mode=str(operating_mode),
                surrogate_library=str(surrogate_library),
                surrogate_domain_min_kg_per_h=domain_min,
                surrogate_domain_max_kg_per_h=domain_max,
                fixed_setpoint_kg_per_h=domain_max,
                allow_soft_extrapolation_below_min=bool(allow_soft_extrapolation_below_min),
                soft_extrapolation_margin_fraction=float(soft_extrapolation_margin_fraction),
            ),
            optimization=OptimizationInputs(
                electrolyzer_power_grid_mw=sorted({max(10.0, electrolyzer_power_mw * 0.8), electrolyzer_power_mw, electrolyzer_power_mw * 1.2}),
                storage_grid_kg_h2=sorted({0.0, storage_kg_h2, max(storage_kg_h2 * 1.5, 1000.0)}),
                module_count_grid=sorted({1, int(module_count), max(int(module_count) + 2, 2)}),
            ),
            renewable_profile=renewable_profile,
            time_step_h=1.0,
        )

    def run_simulation(self, case: CaseInputs, progress_callback=None):
        return self.engine.run(case, progress_callback=progress_callback)

    def run_optimization(self, case: CaseInputs, progress_callback=None):
        return self.optimizer.run(case, progress_callback=progress_callback)

    def run_sensitivity(self, case: CaseInputs, progress_callback=None):
        rows = []
        for i, factor in enumerate([0.8, 0.9, 1.1, 1.2], start=1):
            perturbed = self.build_case(
                scenario_name=case.scenario_name,
                electrolyzer_power_mw=case.electrolyzer.nominal_power_mw * factor,
                module_count=max(int(round(case.electrolyzer.nominal_power_mw * factor / max(case.electrolyzer.module_size_mw, 0.1))), 1),
                storage_enabled=case.storage.enabled,
                storage_kg_h2=case.storage.usable_capacity_kg_h2,
                operating_mode=case.ptmeoh.operating_mode,
                surrogate_library=case.ptmeoh.surrogate_library,
                renewable_peak_power_mw=float(case.renewable_profile['renewable_power_mw'].max()),
                renewable_profile_df=case.renewable_profile,
                electricity_price_usd_per_kwh=case.economic.electricity_price_usd_per_mwh / 1000.0,
                allow_soft_extrapolation_below_min=case.ptmeoh.allow_soft_extrapolation_below_min,
                soft_extrapolation_margin_fraction=case.ptmeoh.soft_extrapolation_margin_fraction,
            )
            sim = self.engine.run(perturbed)
            rows.append({
                'parameter': f'electrolyzer_power_x{factor:.1f}',
                'lcomeoh_usd_per_t_meoh': sim.kpis['lcomeoh_usd_per_t_meoh'],
                'npv_usd': sim.kpis['npv_usd'],
                'annual_methanol_t': sim.kpis['annual_methanol_t'],
            })
            if progress_callback is not None:
                progress_callback('sensitivity', i, 4)
        return pd.DataFrame(rows)
