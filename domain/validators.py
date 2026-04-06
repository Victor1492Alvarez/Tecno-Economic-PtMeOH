from __future__ import annotations
from domain.data_models import CaseInputs

def validate_case_inputs(case: CaseInputs) -> list[str]:
    warnings: list[str] = []
    rp = case.renewable_profile
    if rp.empty or "renewable_power_mw" not in rp.columns:
        warnings.append("Renewable profile is missing or invalid.")
    if (rp.get("renewable_power_mw", 0) < 0).any():
        warnings.append("Renewable profile contains negative power values.")
    if case.electrolyzer.nominal_power_mw <= 0:
        warnings.append("Electrolyzer nominal power must be positive.")
    if case.electrolyzer.module_size_mw <= 0:
        warnings.append("Electrolyzer module size must be positive.")
    if case.storage.enabled and case.storage.usable_capacity_kg_h2 < 0:
        warnings.append("H2 storage capacity cannot be negative.")
    if case.ptmeoh.max_h2_feed_kg_per_h <= 0:
        warnings.append("PtMeOH maximum H2 intake must be positive.")
    if case.ptmeoh.target_h2_feed_kg_per_h > case.ptmeoh.max_h2_feed_kg_per_h:
        warnings.append("Target H2 feed exceeds PtMeOH maximum intake.")
    if case.economic.discount_rate < 0:
        warnings.append("Discount rate cannot be negative.")
    if case.economic.project_years <= 0:
        warnings.append("Project horizon must be positive.")
    return warnings

