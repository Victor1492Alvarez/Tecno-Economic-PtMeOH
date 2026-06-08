from __future__ import annotations

from domain.data_models import CaseInputs
from domain.units import KG_PER_T


class TechnoEconomics:
    def __init__(self, case: CaseInputs):
        self.case = case

    def _crf(self, rate: float, years: int) -> float:
        if years <= 0:
            return 1.0
        if abs(rate) < 1e-12:
            return 1.0 / years
        return rate * (1.0 + rate) ** years / ((1.0 + rate) ** years - 1.0)

    def compute(self, df):
        dt = float(self.case.time_step_h)

        annual_meoh_t = float(df["methanol_t_per_h"].sum() * dt) if "methanol_t_per_h" in df.columns else 0.0
        annual_power_mwh = float(df["total_internal_power_mw"].sum() * dt) if "total_internal_power_mw" in df.columns else 0.0
        annual_downstream_power_mwh = float(df["downstream_aux_power_mw"].sum() * dt) if "downstream_aux_power_mw" in df.columns else 0.0
        annual_h2_t = float(df["h2_produced_kg_per_h"].sum() * dt / KG_PER_T) if "h2_produced_kg_per_h" in df.columns else 0.0

        electrolyzer_capex_usd = (
            self.case.electrolyzer.nominal_power_mw * 1000.0 * self.case.electrolyzer.capex_usd_per_kw
        )
        storage_capex_usd = (
            self.case.storage.usable_capacity_kg_h2 * self.case.storage.capex_usd_per_kg_h2
            if self.case.storage.enabled
            else 0.0
        )
        downstream_capex_usd = 0.15 * electrolyzer_capex_usd
        total_capex_usd = (
            electrolyzer_capex_usd + storage_capex_usd + downstream_capex_usd
        ) * float(self.case.economic.capex_multiplier)

        annual_fixed_opex_usd = (
            electrolyzer_capex_usd * self.case.electrolyzer.fixed_opex_fraction
            + storage_capex_usd * 0.01
            + downstream_capex_usd * 0.03
        )
        annual_electricity_cost_usd = annual_power_mwh * self.case.economic.electricity_price_usd_per_mwh
        annual_co2_t = max(annual_meoh_t * 1.375, 0.0)
        annual_co2_cost_usd = annual_co2_t * self.case.economic.co2_price_usd_per_t

        annual_opex_usd = (
            annual_fixed_opex_usd + annual_electricity_cost_usd + annual_co2_cost_usd
        ) * float(self.case.economic.opex_multiplier)

        crf = self._crf(self.case.economic.discount_rate, self.case.economic.project_years)
        annualized_capex_usd = total_capex_usd * crf

        lcoh_usd_per_t_h2 = (
            (annualized_capex_usd + annual_opex_usd) / annual_h2_t
            if annual_h2_t > 1e-9
            else float("inf")
        )
        lcomeoh_usd_per_t_meoh = (
            (annualized_capex_usd + annual_opex_usd) / annual_meoh_t
            if annual_meoh_t > 1e-9
            else float("inf")
        )

        annual_revenue_usd = annual_meoh_t * self.case.economic.methanol_price_usd_per_t
        annual_cashflow_usd = annual_revenue_usd - annual_opex_usd
        npv_usd = -total_capex_usd
        for year in range(1, self.case.economic.project_years + 1):
            npv_usd += annual_cashflow_usd / ((1.0 + self.case.economic.discount_rate) ** year)

        return {
            "annual_meoh_t": annual_meoh_t,
            "annual_power_mwh": annual_power_mwh,
            "annual_downstream_power_mwh": annual_downstream_power_mwh,
            "total_capex_usd": total_capex_usd,
            "annual_opex_usd": annual_opex_usd,
            "lcoh_usd_per_t_h2": lcoh_usd_per_t_h2,
            "lcomeoh_usd_per_t_meoh": lcomeoh_usd_per_t_meoh,
            "npv_usd": npv_usd,
        }
