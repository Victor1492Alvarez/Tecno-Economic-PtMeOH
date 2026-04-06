from __future__ import annotations
from dataclasses import dataclass

@dataclass
class StorageState:
    soc_kg_h2: float
    tank_in_kg_per_h: float
    tank_out_kg_per_h: float
    tank_full_event: int
    tank_empty_event: int

class H2StorageModel:
    def __init__(self, capacity_kg_h2: float, initial_soc_fraction: float, max_charge_kg_per_h: float | None = None, max_discharge_kg_per_h: float | None = None):
        self.capacity_kg_h2 = max(capacity_kg_h2, 0.0)
        self.soc_kg_h2 = self.capacity_kg_h2 * max(min(initial_soc_fraction, 1.0), 0.0)
        self.max_charge_kg_per_h = max_charge_kg_per_h or max(self.capacity_kg_h2, 1.0)
        self.max_discharge_kg_per_h = max_discharge_kg_per_h or max(self.capacity_kg_h2, 1.0)

    def charge(self, available_kg_per_h: float) -> StorageState:
        room = max(self.capacity_kg_h2 - self.soc_kg_h2, 0.0)
        flow = min(max(available_kg_per_h, 0.0), room, self.max_charge_kg_per_h)
        self.soc_kg_h2 += flow
        return StorageState(self.soc_kg_h2, flow, 0.0, int(room <= 1e-9), 0)

    def discharge(self, demand_kg_per_h: float) -> StorageState:
        flow = min(max(demand_kg_per_h, 0.0), self.soc_kg_h2, self.max_discharge_kg_per_h)
        self.soc_kg_h2 -= flow
        return StorageState(self.soc_kg_h2, 0.0, flow, 0, int(self.soc_kg_h2 <= 1e-9))

