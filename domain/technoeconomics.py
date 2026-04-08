from __future__ import annotations

import pandas as pd

from domain.data_models import CaseInputs
from domain.units import CO2_T_PER_T_MEOH_STOICH, KG_PER_T


class TechnoEconomics:
    def __init__(self, case: CaseInputs):
        self.case = case

    def annualize_capex(self, total_capex_usd: float) -> float:
        r = self.case.economic.discount_rate
        n = self.case.economic.project_years
        if r == 0:
            return total_capex_usd / max(n, 1)
        crf = r * (1 + r) ** n / ((1 + r) ** n - 1)
        return total_capex_usd * crf

    def compute(self, ts: pd.DataFrame) -> dict[str, float]:
        annual_meoh_t = float(ts["methanol_t_per_h"].sum() * self.case.time_step_h)
        annual_h2_t = float(
            ts["h2_produced_kg_per_h"].sum() * self.case.time_step_h / KG_PER_T
        )

        annual_electrolyzer_power_mwh = float(
            ts["power_to_electrolyzer_mw"].sum() * self.case.time_step_h
        )
        annual_downstream_power_mwh = float(
            ts["downstream_aux_power_mw"].sum() * self.case.time_step_h
        )
        annual_power_mwh = annual_electrolyzer_power_mwh + annual_downstream_power_mwh

        electrolyzer_capex = (
            self.case.electrolyzer.nominal_power_mw
            * 1000.0
            * self.case.electrolyzer.capex_usd_per_kw
        )
        storage_capex = (
            self.case.storage.usable_capacity_kg_h2 * self.case.storage.capex_usd_per_kg_h2
            if self.case.storage.enabled
            else 0.0
        )
        total_capex = (electrolyzer_capex + storage_capex) * self.case.economic.capex_multiplier
        annualized_capex = self.annualize_capex(total_capex)

        fixed_opex = (
            electrolyzer_capex
            * self.case.electrolyzer.fixed_opex_fraction
            * self.case.economic.opex_multiplier
        )
        electricity_cost = annual_power_mwh * self.case.economic.electricity_price_usd_per_mwh
        co2_cost = annual_meoh_t * CO2_T_PER_T_MEOH_STOICH * self.case.economic.co2_price_usd_per_t
        annual_opex = fixed_opex + electricity_cost + co2_cost

        methanol_revenue = annual_meoh_t * self.case.economic.methanol_price_usd_per_t

        lcoh = (annualized_capex + annual_opex) / max(annual_h2_t, 1e-9)
        lcomeoh = (annualized_capex + annual_opex) / max(annual_meoh_t, 1e-9)

        npv = (
            sum(
                (methanol_revenue - annual_opex)
                / ((1 + self.case.economic.discount_rate) ** year)
                for year in range(1, self.case.economic.project_years + 1)
            )
            - total_capex
        )

        return {
            "total_capex_usd": total_capex,
            "annualized_capex_usd": annualized_capex,
            "annual_opex_usd": annual_opex,
            "annual_meoh_t": annual_meoh_t,
            "annual_h2_t": annual_h2_t,
            "annual_power_mwh": annual_power_mwh,
            "annual_electrolyzer_power_mwh": annual_electrolyzer_power_mwh,
            "annual_downstream_power_mwh": annual_downstream_power_mwh,
            "electricity_cost_usd": electricity_cost,
            "lcoh_usd_per_t_h2": lcoh,
            "lcomeoh_usd_per_t_meoh": lcomeoh,
            "npv_usd": npv,
            "methanol_revenue_usd": methanol_revenue,
        }
