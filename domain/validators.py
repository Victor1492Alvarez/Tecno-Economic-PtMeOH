from __future__ import annotations

from domain.data_models import CaseInputs


def validate_case_inputs(case: CaseInputs) -> list[str]:
    warnings: list[str] = []

    if case.electrolyzer.nominal_power_mw <= 0:
        warnings.append("Electrolyzer nominal power is non-positive.")
    if case.ptmeoh.target_h2_feed_kg_per_h > case.ptmeoh.max_h2_feed_kg_per_h:
        warnings.append("Target H2 feed exceeds maximum PtMeOH H2 intake.")
    if case.storage.enabled and case.storage.usable_capacity_kg_h2 <= 0:
        warnings.append("H2 storage is enabled but usable capacity is zero or negative.")
    if case.renewable_profile.empty:
        warnings.append("Renewable profile is empty.")
    if "renewable_power_mw" not in case.renewable_profile.columns:
        warnings.append("Renewable profile does not contain 'renewable_power_mw'.")
    if "timestamp" not in case.renewable_profile.columns:
        warnings.append("Renewable profile does not contain 'timestamp'.")

    return warnings
