"""Conditions for generation without a reference trajectory."""

from dataclasses import dataclass
import math
from numbers import Integral


@dataclass(frozen=True)
class GenerationRequest:
    start_road: int
    destination_road: int
    start_alpha: float
    end_alpha: float
    point_count: int
    depart_bin: int

    def __post_init__(self):
        if any(not isinstance(v, Integral) for v in
               (self.start_road, self.destination_road, self.point_count, self.depart_bin)):
            raise ValueError("Road IDs, point count and departure bin must be integers")
        if min(self.start_road, self.destination_road) < 0 or self.point_count < 2:
            raise ValueError("Require nonnegative road IDs and at least two points")
        if not 0 <= self.depart_bin < 288:
            raise ValueError("Departure bin must be between 0 and 287")
        if any(not math.isfinite(a) or not 0 <= a <= 1 for a in (self.start_alpha, self.end_alpha)):
            raise ValueError("Endpoint fractions must be finite and between 0 and 1")
