from __future__ import annotations


class H2StorageModel:
    def __init__(
        self,
        capacity_kg_h2: float,
        initial_soc_fraction: float,
        max_charge_kg_per_h: float | None = None,
        max_discharge_kg_per_h: float | None = None,
    ):
        self.capacity_kg_h2 = float(max(capacity_kg_h2, 0.0))
        self.initial_soc_fraction = float(min(max(initial_soc_fraction, 0.0), 1.0))
        self.max_charge_kg_per_h = float(max_charge_kg_per_h) if max_charge_kg_per_h is not None else self.capacity_kg_h2
        self.max_discharge_kg_per_h = float(max_discharge_kg_per_h) if max_discharge_kg_per_h is not None else self.capacity_kg_h2
        self.soc_kg_h2 = self.capacity_kg_h2 * self.initial_soc_fraction
