from __future__ import annotations

from pathlib import Path
import pandas as pd

from domain.data_models import CaseInputs, SimulationArtifacts
from domain.electrolyzer_model import ElectrolyzerModel
from domain.h2_storage_model import H2StorageModel
from domain.ptmeoh_surrogate import MultiSurrogateManager
from domain.technoeconomics import TechnoEconomics
from domain.validators import validate_case_inputs
from domain.units import KG_PER_T
from infrastructure.logging_utils import configure_logger


class SimulationEngine:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = configure_logger("ptmeoh_tool.simulation")
        self._surrogate_cache: dict[str, MultiSurrogateManager] = {}

    def _get_surrogates(self, library_name: str) -> MultiSurrogateManager:
        if library_name not in self._surrogate_cache:
            self._surrogate_cache[library_name] = MultiSurrogateManager(
                self.project_root,
                library_name,
            )
        return self._surrogate_cache[library_name]

    def run(self, case: CaseInputs) -> SimulationArtifacts:
        warnings = validate_case_inputs(case)

        el = ElectrolyzerModel(
            nominal_power_mw=case.electrolyzer.nominal_power_mw,
            module_size_mw=case.electrolyzer.module_size_mw,
            min_load_fraction=case.electrolyzer.min_load_fraction,
            specific_energy_kwh_per_kg_h2=case.electrolyzer.specific_energy_kwh_per_kg_h2,
        )

        storage = H2StorageModel(
            capacity_kg_h2=case.storage.usable_capacity_kg_h2,
            initial_soc_fraction=case.storage.initial_soc_fraction,
            max_charge_kg_per_h=case.storage.max_charge_kg_per_h,
            max_discharge_kg_per_h=case.storage.max_discharge_kg_per_h,
        )

        surrogates = self._get_surrogates(case.ptmeoh.surrogate_library)

        rows: list[dict] = []
        model_summary_df = pd.DataFrame()

        for _, row in case.renewable_profile.iterrows():
            renewable_power_mw = float(row["renewable_power_mw"])
            timestamp = row.get("timestamp")
            el_state = el.step(renewable_power_mw)

            tank_in = 0.0
            tank_out = 0.0
            tank_full = 0
            tank_empty = 0
            curtailment_h2_kg = 0.0

            if case.ptmeoh.operating_mode == "quasi_base_load":
                target_h2 = min(
                    case.ptmeoh.target_h2_feed_kg_per_h,
                    case.ptmeoh.max_h2_feed_kg_per_h,
                )
                direct_h2 = min(el_state.h2_produced_kg_per_h, target_h2)
                shortage = max(target_h2 - direct_h2, 0.0)
                excess = max(el_state.h2_produced_kg_per_h - direct_h2, 0.0)

                if case.storage.enabled:
                    discharge = storage.discharge(shortage)
                    charge = storage.charge(excess)
                    tank_out = discharge.tank_out_kg_per_h
                    tank_in = charge.tank_in_kg_per_h
                    tank_empty = discharge.tank_empty_event
                    tank_full = charge.tank_full_event
                    curtailment_h2_kg = max(excess - tank_in, 0.0)
                else:
                    curtailment_h2_kg = excess

                actual_h2 = min(
                    direct_h2 + tank_out,
                    case.ptmeoh.max_h2_feed_kg_per_h,
                )

            else:
                raw_available = min(
                    el_state.h2_produced_kg_per_h,
                    case.ptmeoh.max_h2_feed_kg_per_h,
                )
                target_h2 = min(
                    case.ptmeoh.target_h2_feed_kg_per_h,
                    case.ptmeoh.max_h2_feed_kg_per_h,
                )

                if case.storage.enabled and raw_available < target_h2:
                    discharge = storage.discharge(target_h2 - raw_available)
                    tank_out = discharge.tank_out_kg_per_h
                    tank_empty = discharge.tank_empty_event
                    actual_h2 = min(
                        raw_available + tank_out,
                        case.ptmeoh.max_h2_feed_kg_per_h,
                    )
                else:
                    actual_h2 = raw_available

                if case.storage.enabled and raw_available > target_h2:
                    charge = storage.charge(raw_available - target_h2)
                    tank_in = charge.tank_in_kg_per_h
                    tank_full = charge.tank_full_event
                    curtailment_h2_kg = max(raw_available - target_h2 - tank_in, 0.0)
                    actual_h2 = min(target_h2, case.ptmeoh.max_h2_feed_kg_per_h)

            unmet_h2 = max(case.ptmeoh.target_h2_feed_kg_per_h - actual_h2, 0.0)
            surrogate_outputs = surrogates.predict_all(actual_h2)
            model_summary_df = surrogate_outputs["model_summary_df"]

            methanol_t_per_h = max(
                actual_h2 / KG_PER_T * case.ptmeoh.methanol_yield_t_meoh_per_t_h2,
                0.0,
            )

            if "Model_Prod_MeOH" in model_summary_df["model_name"].tolist():
                prod_row = model_summary_df.loc[
                    model_summary_df["model_name"] == "Model_Prod_MeOH"
                ]
                if not prod_row.empty:
                    methanol_t_per_h = max(float(prod_row["prediction"].iloc[0]), 0.0)

            aux_power_from_models_w = (
                float(
                    model_summary_df.loc[
                        model_summary_df["model_name"].str.contains(
                            "Model_Power_",
                            regex=False,
                        ),
                        "prediction",
                    ]
                    .clip(lower=0)
                    .sum()
                )
                if not model_summary_df.empty
                else 0.0
            )

            record = {
                "timestamp": timestamp,
                "renewable_power_mw": renewable_power_mw,
                "power_to_electrolyzer_mw": el_state.power_to_electrolyzer_mw,
                "module_count_online": el_state.module_count_online,
                "h2_produced_kg_per_h": el_state.h2_produced_kg_per_h,
                "curtailed_power_mw": el_state.curtailed_power_mw,
                "curtailed_h2_kg_per_h": curtailment_h2_kg,
                "tank_soc_kg_h2": storage.soc_kg_h2,
                "tank_in_kg_per_h": tank_in,
                "tank_out_kg_per_h": tank_out,
                "tank_full_event": tank_full,
                "tank_empty_event": tank_empty,
                "h2_to_ptmeoh_kg_per_h": actual_h2,
                "unmet_h2_kg_per_h": unmet_h2,
                "methanol_t_per_h": methanol_t_per_h,
                "aux_power_from_models_w": aux_power_from_models_w,
                "surrogate_all_models_in_domain": int(
                    bool(surrogate_outputs["all_models_in_domain"])
                ),
                "surrogate_runtime_models_count": int(
                    surrogate_outputs["runtime_models_count"]
                ),
                "surrogate_total_models_count": int(
                    surrogate_outputs["total_models_count"]
                ),
            }

            for key, value in surrogate_outputs.items():
                if key == "model_summary_df":
                    continue
                if key not in record:
                    record[key] = value

            rows.append(record)

        df = pd.DataFrame(rows)
        te = TechnoEconomics(case)
        econ = te.compute(df)

        kpis = {
            "annual_methanol_t": econ["annual_meoh_t"],
            "electrolyzer_full_load_hours_h": float(
                df["power_to_electrolyzer_mw"].sum()
                * case.time_step_h
                / max(case.electrolyzer.nominal_power_mw, 1e-9)
            ),
            "ptmeoh_utilization_factor": float(
                df["h2_to_ptmeoh_kg_per_h"].mean()
                / max(case.ptmeoh.max_h2_feed_kg_per_h, 1e-9)
            ),
            "renewable_utilization_fraction": float(
                df["power_to_electrolyzer_mw"].sum()
                / max(df["renewable_power_mw"].sum(), 1e-9)
            ),
            "h2_not_supplied_t": float(
                df["unmet_h2_kg_per_h"].sum() * case.time_step_h / KG_PER_T
            ),
            "curtailment_fraction": float(
                df["curtailed_power_mw"].sum()
                / max(df["renewable_power_mw"].sum(), 1e-9)
            ),
            "total_capex_usd": econ["total_capex_usd"],
            "annual_opex_usd": econ["annual_opex_usd"],
            "lcoh_usd_per_t_h2": econ["lcoh_usd_per_t_h2"],
            "lcomeoh_usd_per_t_meoh": econ["lcomeoh_usd_per_t_meoh"],
            "npv_usd": econ["npv_usd"],
            "surrogate_out_of_domain_fraction": float(
                1.0 - df["surrogate_all_models_in_domain"].mean()
            ),
            "tank_empty_hours": float(df["tank_empty_event"].sum() * case.time_step_h),
            "tank_full_hours": float(df["tank_full_event"].sum() * case.time_step_h),
            "curtailed_hours": float(
                (df["curtailed_power_mw"] > 0).sum() * case.time_step_h
            ),
            "runtime_models_fraction": float(
                df["surrogate_runtime_models_count"].max()
                / max(df["surrogate_total_models_count"].max(), 1)
            ),
        }

        unmet_fraction = float(
            df["unmet_h2_kg_per_h"].sum()
            / max(
                df["h2_to_ptmeoh_kg_per_h"].sum()
                + df["unmet_h2_kg_per_h"].sum(),
                1e-9,
            )
        )

        if kpis["surrogate_out_of_domain_fraction"] > 0:
            warnings.append(
                "Selected operating range exceeds the surrogate validity domain during part of the simulated year."
            )
        if unmet_fraction > case.ptmeoh.unmet_h2_warning_threshold:
            warnings.append("Unmet H2 exceeds the configured warning threshold.")
        if kpis["curtailment_fraction"] > case.ptmeoh.curtailment_warning_threshold:
            warnings.append(
                "Renewable curtailment exceeds the configured warning threshold."
            )
        if kpis["ptmeoh_utilization_factor"] < case.ptmeoh.utilization_warning_threshold:
            warnings.append(
                "PtMeOH utilization is below the configured warning threshold."
            )
        if kpis["runtime_models_fraction"] < 1.0:
            warnings.append(
                "At least one surrogate model package is incomplete; fallback mock predictions are being used for missing runtime artifacts."
            )

        surrogate_info = {
            "library_name": case.ptmeoh.surrogate_library,
            "models_detected": int(df["surrogate_total_models_count"].max())
            if not df.empty
            else 0,
            "runtime_models_detected": int(df["surrogate_runtime_models_count"].max())
            if not df.empty
            else 0,
        }

        return SimulationArtifacts(
            time_series=df,
            kpis=kpis,
            warnings=warnings,
            surrogate_info=surrogate_info,
            model_summary=model_summary_df,
        )
