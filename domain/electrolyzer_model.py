from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class ElectrolyzerState:
    power_to_electrolyzer_mw: float
    module_count_online: int
    h2_produced_kg_per_h: float


class ElectrolyzerModel:
    def __init__(
        self,
        nominal_power_mw: float,
        module_size_mw: float,
        min_load_fraction: float,
        specific_energy_kwh_per_kg_h2: float,
    ):
        self.nominal_power_mw = float(max(nominal_power_mw, 0.0))
        self.module_size_mw = float(max(module_size_mw, 1e-9))
        self.min_load_fraction = float(max(min_load_fraction, 0.0))
        self.specific_energy_kwh_per_kg_h2 = float(max(specific_energy_kwh_per_kg_h2, 1e-9))
        self.total_modules = max(1, int(round(self.nominal_power_mw / self.module_size_mw)))

    def step(self, available_power_mw: float) -> ElectrolyzerState:
        available_power_mw = float(max(available_power_mw, 0.0))
        min_power_mw = self.nominal_power_mw * self.min_load_fraction

        if available_power_mw < min_power_mw:
            return ElectrolyzerState(
                power_to_electrolyzer_mw=0.0,
                module_count_online=0,
                h2_produced_kg_per_h=0.0,
            )

        power_to_electrolyzer_mw = min(available_power_mw, self.nominal_power_mw)
        module_count_online = min(
            self.total_modules,
            max(1, int(math.ceil(power_to_electrolyzer_mw / self.module_size_mw - 1e-12))),
        )

        h2_produced_kg_per_h = power_to_electrolyzer_mw * 1000.0 / self.specific_energy_kwh_per_kg_h2

        return ElectrolyzerState(
            power_to_electrolyzer_mw=power_to_electrolyzer_mw,
            module_count_online=module_count_online,
            h2_produced_kg_per_h=h2_produced_kg_per_h,
        )
