from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from domain.data_models import CaseInputs, SimulationArtifacts
from domain.ptmeoh_surrogate import MultiSurrogateManager

ProgressCallback = Callable[[str, int, int], None] | None

class SimulationEngine:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger('ptmeoh_tool.simulation')

    def _emit_progress(self, callback: ProgressCallback, stage: str, current: int, total: int) -> None:
        if callback is not None:
            callback(stage, current, total)

    def _default_profile(self, peak_power_mw: float, hours: int = 8760) -> pd.DataFrame:
        idx = pd.date_range('2025-01-01', periods=hours, freq='h')
        base = 0.55 + 0.25 * np.sin(np.arange(hours) * 2 * np.pi / 24.0) + 0.15 * np.sin(np.arange(hours) * 2 * np.pi / (24.0 * 30.0))
        noise = 0.05 * np.sin(np.arange(hours) * 2 * np.pi / (24.0 * 7.0))
        renewable = np.clip((base + noise), 0.0, 1.0) * peak_power_mw
        return pd.DataFrame({'timestamp': idx, 'renewable_power_mw': renewable})

    def run(self, case: CaseInputs, progress_callback: ProgressCallback = None) -> SimulationArtifacts:
        profile = case.renewable_profile.copy() if case.renewable_profile is not None and not case.renewable_profile.empty else self._default_profile(case.electrolyzer.nominal_power_mw * 1.75)
        profile = profile.sort_values('timestamp').reset_index(drop=True)
        if 'timestamp' not in profile.columns:
            profile['timestamp'] = pd.date_range('2025-01-01', periods=len(profile), freq='h')
        if 'renewable_power_mw' not in profile.columns:
            raise ValueError('renewable_profile must contain renewable_power_mw')

        manager = MultiSurrogateManager(self.project_root, case.ptmeoh.surrogate_library)
        effective_min = float(manager.domain_min)
        effective_max = float(manager.domain_max)
        fixed_setpoint = float(effective_max)
        case.ptmeoh.surrogate_domain_min_kg_per_h = effective_min
        case.ptmeoh.surrogate_domain_max_kg_per_h = effective_max
        case.ptmeoh.fixed_setpoint_kg_per_h = fixed_setpoint

        soc = case.storage.usable_capacity_kg_h2 * case.storage.initial_soc_fraction if case.storage.enabled else 0.0
        storage_cap = case.storage.usable_capacity_kg_h2 if case.storage.enabled else 0.0
        max_charge = case.storage.max_charge_kg_per_h or storage_cap or 1e12
        max_discharge = case.storage.max_discharge_kg_per_h or storage_cap or 1e12

        rows = []
        model_summaries = []
        total_hours = len(profile)
        self._emit_progress(progress_callback, 'simulation', 0, max(total_hours, 1))

        spec_kwh_per_kg = max(case.electrolyzer.specific_energy_kwh_per_kg_h2, 1e-6)
        max_h2_from_nominal = case.electrolyzer.nominal_power_mw * 1000.0 / spec_kwh_per_kg
        aux_power_per_kg = 0.02

        for i, row in profile.iterrows():
            renewable_power_mw = float(max(row['renewable_power_mw'], 0.0))
            power_to_electrolyzer_mw = min(renewable_power_mw, case.electrolyzer.nominal_power_mw)
            min_load_mw = case.electrolyzer.nominal_power_mw * case.electrolyzer.min_load_fraction
            electrolyzer_on = power_to_electrolyzer_mw >= min_load_mw
            if not electrolyzer_on:
                power_to_electrolyzer_mw = 0.0
            h2_produced = min(power_to_electrolyzer_mw * 1000.0 / spec_kwh_per_kg, max_h2_from_nominal)
            available_h2 = h2_produced
            h2_to_storage = 0.0
            h2_from_storage = 0.0
            h2_spilled = 0.0
            requested_h2 = fixed_setpoint
            feed_candidate = min(available_h2, requested_h2)
            shortage_before_storage = max(requested_h2 - feed_candidate, 0.0)

            if case.storage.enabled and shortage_before_storage > 0:
                can_discharge = min(shortage_before_storage, soc, max_discharge)
                h2_from_storage = can_discharge
                soc -= can_discharge
                feed_candidate += can_discharge

            if feed_candidate > fixed_setpoint:
                feed_candidate = fixed_setpoint

            excess_after_feed = max(available_h2 - min(available_h2, fixed_setpoint), 0.0)
            if case.storage.enabled and excess_after_feed > 0:
                free_capacity = max(storage_cap - soc, 0.0)
                charge = min(excess_after_feed, free_capacity, max_charge)
                h2_to_storage = charge
                soc += charge
                h2_spilled = max(excess_after_feed - charge, 0.0)
            else:
                h2_spilled = excess_after_feed

            surrogate = manager.predict_all(
                feed_candidate,
                allow_soft_extrapolation_below_min=case.ptmeoh.allow_soft_extrapolation_below_min,
                soft_extrapolation_margin_fraction=case.ptmeoh.soft_extrapolation_margin_fraction,
            )
            methanol_t_per_h = float(surrogate.get('MeOHProd', 0.0))
            if methanol_t_per_h <= 0.0:
                methanol_t_per_h = (feed_candidate / 1000.0) * case.ptmeoh.methanol_yield_t_meoh_per_t_h2
            downstream_aux_power_mw = max(feed_candidate * aux_power_per_kg / 1000.0, 0.0)
            renewable_used_mw = power_to_electrolyzer_mw + downstream_aux_power_mw
            curtailed_power_mw = max(renewable_power_mw - renewable_used_mw, 0.0)
            ptmeoh_setpoint_gap = max(fixed_setpoint - feed_candidate, 0.0)
            power_deficit_mw = max(renewable_used_mw - renewable_power_mw, 0.0)
            tank_empty = 1 if case.storage.enabled and soc <= 1e-9 else 0
            tank_full = 1 if case.storage.enabled and soc >= max(storage_cap - 1e-9, 0.0) and storage_cap > 0 else 0

            rows.append({
                'timestamp': row['timestamp'],
                'renewable_power_mw': renewable_power_mw,
                'renewable_used_mw': renewable_used_mw,
                'power_to_electrolyzer_mw': power_to_electrolyzer_mw,
                'downstream_aux_power_mw': downstream_aux_power_mw,
                'curtailed_power_mw': curtailed_power_mw,
                'h2_produced_kg_per_h': h2_produced,
                'h2_to_ptmeoh_kg_per_h': feed_candidate,
                'h2_to_storage_kg_per_h': h2_to_storage,
                'h2_from_storage_kg_per_h': h2_from_storage,
                'h2_spilled_kg_per_h': h2_spilled,
                'tank_soc_kg_h2': soc,
                'ptmeoh_setpoint_kg_per_h': fixed_setpoint,
                'ptmeoh_setpoint_gap_kg_per_h': ptmeoh_setpoint_gap,
                'unmet_h2_kg_per_h': ptmeoh_setpoint_gap,
                'methanol_t_per_h': methanol_t_per_h,
                'surrogate_eval_mode': surrogate.get('eval_mode', 'in_domain'),
                'surrogate_all_models_in_domain': int(bool(surrogate.get('all_models_in_domain', True))),
                'surrogate_soft_extrapolated': int(bool(surrogate.get('soft_extrapolated_any', False))),
                'surrogate_hard_out_of_range': int(bool(surrogate.get('hard_out_of_range_any', False))),
                'power_deficit_mw': power_deficit_mw,
            })
            model_df = surrogate.get('model_summary_df', pd.DataFrame())
            if isinstance(model_df, pd.DataFrame) and not model_df.empty:
                model_df = model_df.copy()
                model_df['timestamp'] = row['timestamp']
                model_summaries.append(model_df)
            self._emit_progress(progress_callback, 'simulation', i + 1, max(total_hours, 1))

        ts = pd.DataFrame(rows)
        annual_methanol_t = float(ts['methanol_t_per_h'].sum())
        annual_h2_to_ptmeoh_t = float(ts['h2_to_ptmeoh_kg_per_h'].sum() / 1000.0)
        annual_setpoint_gap_t = float(ts['ptmeoh_setpoint_gap_kg_per_h'].sum() / 1000.0)
        electricity_cost = float(ts['renewable_used_mw'].sum() * case.economic.electricity_price_usd_per_mwh)
        electrolyzer_capex = case.electrolyzer.nominal_power_mw * 1000.0 * case.electrolyzer.capex_usd_per_kw * case.economic.capex_multiplier
        storage_capex = case.storage.usable_capacity_kg_h2 * case.storage.capex_usd_per_kg_h2 if case.storage.enabled else 0.0
        total_capex = electrolyzer_capex + storage_capex
        fixed_opex = total_capex * case.electrolyzer.fixed_opex_fraction * case.economic.opex_multiplier
        co2_cost = annual_methanol_t * 0.1 * case.economic.co2_price_usd_per_t
        revenue = annual_methanol_t * case.economic.methanol_price_usd_per_t
        annual_cash = revenue - electricity_cost - fixed_opex - co2_cost
        discounted = sum(annual_cash / ((1 + case.economic.discount_rate) ** y) for y in range(1, case.economic.project_years + 1))
        npv = discounted - total_capex
        lcomeoh = (electricity_cost + fixed_opex + co2_cost + total_capex / max(case.economic.project_years, 1)) / max(annual_methanol_t, 1e-9)

        kpis = {
            'annual_methanol_t': annual_methanol_t,
            'electrolyzer_full_load_hours_h': float(ts['power_to_electrolyzer_mw'].sum() / max(case.electrolyzer.nominal_power_mw, 1e-9)),
            'ptmeoh_utilization_factor': float(ts['h2_to_ptmeoh_kg_per_h'].mean() / max(fixed_setpoint, 1e-9)),
            'renewable_utilization_fraction': float(ts['renewable_used_mw'].sum() / max(ts['renewable_power_mw'].sum(), 1e-9)),
            'lcomeoh_usd_per_t_meoh': lcomeoh,
            'npv_usd': npv,
            'surrogate_out_of_domain_fraction': float((1 - ts['surrogate_all_models_in_domain']).mean()),
            'soft_extrapolated_fraction': float(ts['surrogate_soft_extrapolated'].mean()),
            'hard_out_of_range_fraction': float(ts['surrogate_hard_out_of_range'].mean()),
            'curtailment_fraction': float(ts['curtailed_power_mw'].sum() / max(ts['renewable_power_mw'].sum(), 1e-9)),
            'h2_not_supplied_t': annual_setpoint_gap_t,
            'tank_empty_hours_h': float(ts['tank_soc_kg_h2'].le(1e-9).sum()),
            'tank_full_hours_h': float(ts['tank_soc_kg_h2'].ge(max(storage_cap - 1e-9, 0.0)).sum()) if storage_cap > 0 else 0.0,
            'runtime_models_fraction': 1.0,
            'annual_total_electricity_mwh': float(ts['renewable_used_mw'].sum()),
            'total_capex_usd': float(total_capex),
            'fixed_setpoint_kg_per_h': float(fixed_setpoint),
            'surrogate_domain_min_kg_per_h': float(effective_min),
            'surrogate_domain_max_kg_per_h': float(effective_max),
        }

        warnings = []
        if kpis['soft_extrapolated_fraction'] > 0:
            warnings.append(f"Soft extrapolation was used during {100.0 * kpis['soft_extrapolated_fraction']:.2f}% of the simulated hours. These timesteps were evaluated below the validated surrogate minimum and should be interpreted with caution.")
        if kpis['hard_out_of_range_fraction'] > 0:
            warnings.append(f"Hard out-of-range operation occurred during {100.0 * kpis['hard_out_of_range_fraction']:.2f}% of the simulated hours. For these hours, the PtMeOH response was bounded conservatively at the surrogate edge.")
        if kpis['h2_not_supplied_t'] / max(annual_h2_to_ptmeoh_t + annual_setpoint_gap_t, 1e-9) > case.ptmeoh.unmet_h2_warning_threshold:
            warnings.append('The PtMeOH set point gap exceeds the warning threshold; the renewable-plus-storage system frequently fails to sustain the fixed PtMeOH target.')
        if kpis['curtailment_fraction'] > case.ptmeoh.curtailment_warning_threshold:
            warnings.append('Curtailment fraction exceeds the warning threshold.')
        if kpis['ptmeoh_utilization_factor'] < case.ptmeoh.utilization_warning_threshold:
            warnings.append('PtMeOH utilization is below the warning threshold.')

        surrogate_info = {
            'library': case.ptmeoh.surrogate_library,
            'effective_domain_min_kg_per_h': effective_min,
            'effective_domain_max_kg_per_h': effective_max,
            'fixed_setpoint_kg_per_h': fixed_setpoint,
            'allow_soft_extrapolation_below_min': case.ptmeoh.allow_soft_extrapolation_below_min,
            'soft_extrapolation_margin_fraction': case.ptmeoh.soft_extrapolation_margin_fraction,
        }
        model_summary = pd.concat(model_summaries, ignore_index=True) if model_summaries else pd.DataFrame()
        return SimulationArtifacts(time_series=ts, kpis=kpis, warnings=warnings, surrogate_info=surrogate_info, model_summary=model_summary)
