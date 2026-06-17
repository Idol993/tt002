import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import copy


class DualModuleWrapper(nn.Module):
    def __init__(self, module_a: nn.Module, module_b: nn.Module, name_a: str = "syntax", name_b: str = "semantics"):
        super().__init__()
        self.module_a = module_a
        self.module_b = module_b
        self.name_a = name_a
        self.name_b = name_b
        self._swapped = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._swapped:
            h = self.module_b(x)
            return self.module_a(h)
        h = self.module_a(x)
        return self.module_b(h)

    def forward_separate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_a = self.module_a(x)
        out_b = self.module_b(x)
        return out_a, out_b

    def forward_with_intermediate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._swapped:
            inter = self.module_b(x)
            out = self.module_a(inter)
            return out, inter
        inter = self.module_a(x)
        out = self.module_b(inter)
        return out, inter

    @property
    def is_swapped(self) -> bool:
        return self._swapped

    def swap_parameters(self) -> Dict[str, torch.Tensor]:
        state_a = {k: v.clone() for k, v in self.module_a.state_dict().items()}
        state_b = {k: v.clone() for k, v in self.module_b.state_dict().items()}

        self.module_a.load_state_dict(state_b)
        self.module_b.load_state_dict(state_a)

        self._swapped = not self._swapped

        return {"pre_swap_a": state_a, "pre_swap_b": state_b}

    def swap_optimizer_states(self, opt_a: torch.optim.Optimizer, opt_b: torch.optim.Optimizer) -> Dict:
        state_a_copy = copy.deepcopy(opt_a.state_dict())
        state_b_copy = copy.deepcopy(opt_b.state_dict())

        opt_a.load_state_dict(state_b_copy)
        opt_b.load_state_dict(state_a_copy)

        return {"pre_swap_opt_a": state_a_copy, "pre_swap_opt_b": state_b_copy}

    def get_module_params(self, module_key: str) -> List[nn.Parameter]:
        module = self.module_a if module_key == "a" else self.module_b
        return list(module.parameters())

    def get_all_params(self) -> List[nn.Parameter]:
        return list(self.module_a.parameters()) + list(self.module_b.parameters())
