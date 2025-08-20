from dataclasses import dataclass, field
from torch import Tensor


@dataclass
class Option:
    loss: Tensor | float = 0.0
    perturbation: Tensor | int = 0
    perturb_idx: Tensor | int = 0
    options: list = field(default_factory=list)

    def apply(self, parameter):
        parameter.flatten()[self.perturb_idx] += self.perturbation

    def __lt__(self, other):
        return self.loss < other.loss
