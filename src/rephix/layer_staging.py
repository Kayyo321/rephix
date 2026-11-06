"""Stage the state needed for one local layer observation at a time."""

from __future__ import annotations

import copy
from itertools import chain


class LayerStager:
    """Keep the last requested group of probe modules on the oracle device."""

    def __init__(self, torch, device):
        self._torch = torch
        self._device = device
        self._sources = ()
        self._copies = ()
        self.moves = 0
        self.peak_state_bytes = 0

    def clear(self):
        """Release the staged modules without resetting the accounting."""
        self._copies = ()
        self._sources = ()

    def get(self, *modules):
        """Return isolated probe modules, preserving their registered state aliases."""
        if len(modules) == len(self._sources) and all(
            module is source for module, source in zip(modules, self._sources)
        ):
            return self._copies
        self.clear()

        # Seed deepcopy with the moved tensors. Copying whole modules before
        # moving them would allocate a second copy of their state on the host.
        memo = {}
        state_bytes = 0
        parameter_type = self._torch.nn.Parameter
        try:
            for module in modules:
                for tensor in chain(module.parameters(), module.buffers()):
                    key = id(tensor)
                    if key in memo:
                        continue
                    moved = tensor.detach().to(device=self._device, copy=True)
                    if isinstance(tensor, parameter_type):
                        moved = parameter_type(moved, requires_grad=tensor.requires_grad)
                    else:
                        moved.requires_grad_(tensor.requires_grad)
                    memo[key] = moved
                    state_bytes += moved.numel() * moved.element_size()
            copies = copy.deepcopy(modules, memo)
        finally:
            memo.clear()

        self._sources = modules
        self._copies = copies
        self.moves += 1
        self.peak_state_bytes = max(self.peak_state_bytes, state_bytes)
        return copies
