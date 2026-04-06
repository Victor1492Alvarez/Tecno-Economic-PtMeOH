from __future__ import annotations
from dataclasses import dataclass
from domain.units import KWH_PER_MWH

@dataclass
class ElectrolyzerState:
    power_to_electrolyzer_mw: float
    h2_produced_kg_per_h: float
    curtailed_power_mw: float
    module_count_online: int

class ElectrolyzerModel:
    def __init__(self, nominal_power_mw: float, module_size_mw: float, min_load_fraction: float, specific_energy_kwh_per_kg_h2: float):
        self.nominal_power_mw = nominal_power_mw
        self.module_size_mw = module_size_mw
        self.min_load_fraction = min_load_fraction
        self.specific_energy_kwh_per_kg_h2 = specific_energy_kwh_per_kg_h2

    def step(self, renewable_power_mw: float) -> ElectrolyzerState:
        available = max(renewable_power_mw, 0.0)
        power = min(available, self.nominal_power_mw)
        min_stable_power = self.nominal_power_mw * self.min_load_fraction
        if 0.0 < power < min_stable_power:
            power = 0.0
        modules_online = int(power // self.module_size_mw) if self.module_size_mw > 0 else 0
        if power > 0 and modules_online == 0:
            modules_online = 1
        h2_kg_per_h = power * KWH_PER_MWH / max(self.specific_energy_kwh_per_kg_h2, 1e-9)
        curtailed = max(available - power, 0.0)
        return ElectrolyzerState(power_to_electrolyzer_mw=power, h2_produced_kg_per_h=h2_kg_per_h, curtailed_power_mw=curtailed, module_count_online=modules_online)

