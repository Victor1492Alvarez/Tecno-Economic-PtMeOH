from __future__ import annotations
import pandas as pd
from domain.data_models import CaseInputs, ScenarioEconomicInputs
from domain.simulation_engine import SimulationEngine

class SensitivityAnalyzer:
    def __init__(self, engine: SimulationEngine):
        self.engine = engine

    def run(self, case: CaseInputs) -> pd.DataFrame:
        base = {
            "electricity_price_usd_per_mwh": case.economic.electricity_price_usd_per_mwh,
            "co2_price_usd_per_t": case.economic.co2_price_usd_per_t,
            "methanol_price_usd_per_t": case.economic.methanol_price_usd_per_t,
            "discount_rate": case.economic.discount_rate,
        }
        rows = []
        for parameter, value in base.items():
            for shift in (-0.20, 0.20):
                modified_dict = {**case.economic.__dict__}
                modified_dict[parameter] = value * (1 + shift)
                modified = ScenarioEconomicInputs(**modified_dict)
                modified_case = CaseInputs(
                    case_name=f"sens_{parameter}_{shift:+.2f}",
                    scenario_name=case.scenario_name,
                    economic=modified,
                    electrolyzer=case.electrolyzer,
                    storage=case.storage,
                    ptmeoh=case.ptmeoh,
                    optimization=case.optimization,
                    renewable_profile=case.renewable_profile,
                    time_step_h=case.time_step_h,
                )
                sim = self.engine.run(modified_case)
                rows.append({
                    "parameter": parameter,
                    "shift_fraction": shift,
                    "lcomeoh_usd_per_t_meoh": sim.kpis["lcomeoh_usd_per_t_meoh"],
                    "npv_usd": sim.kpis["npv_usd"],
                })
        return pd.DataFrame(rows)

