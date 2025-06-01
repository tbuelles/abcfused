from typing import Dict, Any
import torch
from copy import deepcopy


def clone_data(
    data: Dict[str, Any],
    retain_grad: bool = True,
    device=None,
) -> Dict[str, Any]:
    cloned = {}
    for k, v in data.items():
        if not isinstance(v, torch.Tensor):
            cloned[k] = deepcopy(v)
            continue

        # torch.Tensor
        if v.requires_grad:
            v_clone = v.detach().clone().requires_grad_()
            if retain_grad:
                v_clone.retain_grad()
            cloned[k] = v_clone
        else:
            cloned[k] = v.clone()

        if device is not None:
            cloned[k] = cloned[k].to(device)

    return cloned