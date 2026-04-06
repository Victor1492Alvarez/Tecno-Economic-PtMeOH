from __future__ import annotations
import itertools
import pandas as pd
from domain.data_models import CaseInputs, ElectrolyzerInputs, GridSearchArtifacts, PtMeOHInputs, StorageInputs
from domain.simulation_engine import SimulationEngine

class GridOptimizer:
    def __init__(self, engine: SimulationEngine):
        self.engine = engine

    def run(self, base_case: CaseInputs) -> GridSearchArtifacts:
        rows: list[dict] = []
        for power_mw, storage_kg, target_h2, module_count in itertools.product(
            base_case.optimization.electrolyzer_power_grid_mw,
            base_case.optimization.storage_grid_kg_h2,
            base_case.optimization.target_h2_grid_kg_per_h,
            base_case.optimization.module_count_grid,
        ):
            case = CaseInputs(
                case_name=f"P{power_mw}_S{storage_kg}_T{target_h2}_M{module_count}",
                scenario_name=base_case.scenario_name,
                economic=base_case.economic,
                electrolyzer=ElectrolyzerInputs(
                    nominal_power_mw=power_mw,
                    module_size_mw=max(power_mw / max(module_count, 1), 0.1),
                    min_load_fraction=base_case.electrolyzer.min_load_fraction,
                    specific_energy_kwh_per_kg_h2=base_case.electrolyzer.specific_energy_kwh_per_kg_h2,
                    capex_usd_per_kw=base_case.electrolyzer.capex_usd_per_kw,
                    fixed_opex_fraction=base_case.electrolyzer.fixed_opex_fraction,
                ),
                storage=StorageInputs(
                    enabled=base_case.storage.enabled,
                    usable_capacity_kg_h2=storage_kg,
                    initial_soc_fraction=base_case.storage.initial_soc_fraction,
                    max_charge_kg_per_h=base_case.storage.max_charge_kg_per_h,
                    max_discharge_kg_per_h=base_case.storage.max_discharge_kg_per_h,
                    capex_usd_per_kg_h2=base_case.storage.capex_usd_per_kg_h2,
                ),
                ptmeoh=PtMeOHInputs(
                    operating_mode=base_case.ptmeoh.operating_mode,
                    surrogate_library=base_case.ptmeoh.surrogate_library,
                    target_h2_feed_kg_per_h=target_h2,
                    max_h2_feed_kg_per_h=base_case.ptmeoh.max_h2_feed_kg_per_h,
                    methanol_yield_t_meoh_per_t_h2=base_case.ptmeoh.methanol_yield_t_meoh_per_t_h2,
                    unmet_h2_warning_threshold=base_case.ptmeoh.unmet_h2_warning_threshold,
                    curtailment_warning_threshold=base_case.ptmeoh.curtailment_warning_threshold,
                    utilization_warning_threshold=base_case.ptmeoh.utilization_warning_threshold,
                ),
                optimization=base_case.optimization,
                renewable_profile=base_case.renewable_profile,
                time_step_h=base_case.time_step_h,
            )
            sim = self.engine.run(case)
            unmet_fraction = sim.kpis["h2_not_supplied_t"] / max(sim.kpis["annual_methanol_t"], 1e-9)
            rows.append({
                "case_name": case.case_name,
                "electrolyzer_power_mw": power_mw,
                "module_count": module_count,
                "module_size_mw": case.electrolyzer.module_size_mw,
                "storage_kg_h2": storage_kg,
                "target_h2_kg_per_h": target_h2,
                **sim.kpis,
                "warning_count": len(sim.warnings),
                "feasible": int(
                    sim.kpis["ptmeoh_utilization_factor"] >= base_case.optimization.min_ptmeoh_utilization
                    and unmet_fraction <= base_case.optimization.max_unmet_h2_fraction
                    and sim.kpis["curtailment_fraction"] <= base_case.optimization.max_curtailment_fraction
                    and sim.kpis["surrogate_out_of_domain_fraction"] <= 0.0
                )
            })
        results = pd.DataFrame(rows)
        feasible = results[results["feasible"] == 1].copy()
        if feasible.empty:
            feasible = results.copy()
        best = feasible.sort_values(["lcomeoh_usd_per_t_meoh", "warning_count", "total_capex_usd"], ascending=[True, True, True]).iloc
        return GridSearchArtifacts(results=results, feasible_results=feasible, best_row=best)

