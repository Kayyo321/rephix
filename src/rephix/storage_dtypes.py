"""Restore parameter storage precision from safetensors metadata."""

from pathlib import Path

from .weight_audit import _inventory


def _dtype_names(torch):
    return {
        "F16": str(torch.float16),
        "BF16": str(torch.bfloat16),
        "F32": str(torch.float32),
        "F64": str(torch.float64),
    }


def source_parameter_dtypes(torch, source, candidate, aliases, *, unstored_names=()):
    """Resolve every unique parameter's original storage dtype without reading values."""
    entries = _inventory(Path(source))
    if not isinstance(aliases, dict) or any(
        not isinstance(name, str) or not isinstance(stored, str) for name, stored in aliases.items()
    ):
        raise ValueError("source parameter aliases must map parameter names to stored names")
    unstored_names = set(unstored_names)
    names = _dtype_names(torch)
    groups = {}
    for name, parameter in candidate.named_parameters(remove_duplicate=False):
        groups.setdefault(id(parameter), []).append((name, parameter))
    result = {}
    for tied in groups.values():
        canonical, parameter = tied[0]
        observed = set()
        for name, _ in tied:
            stored = aliases.get(name, name)
            entry = entries.get(stored)
            if entry is None:
                continue
            if entry.shape != tuple(parameter.shape):
                raise ValueError(f"{name}: source parameter shape disagrees with the scaffold")
            if entry.dtype not in names:
                raise ValueError(f"{stored}: unsupported source parameter dtype {entry.dtype}")
            observed.add(names[entry.dtype])
        if not observed:
            if all(name in unstored_names for name, _ in tied):
                continue
            raise ValueError(f"{canonical}: source parameter dtype metadata is missing")
        if len(observed) != 1:
            raise ValueError(f"{canonical}: tied source parameters have conflicting dtypes")
        result[canonical] = observed.pop()
    return result


def apply_parameter_dtypes(torch, model, dtypes):
    """Cast each selected Parameter once, preserving ties and validating before mutation."""
    if not isinstance(dtypes, dict):
        raise ValueError("parameter dtypes must be a mapping")
    supported = {
        str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    }
    parameters = dict(model.named_parameters(remove_duplicate=False))
    planned = {}
    for name, dtype in dtypes.items():
        if name not in parameters:
            raise ValueError(f"{name}: dtype metadata names an unknown parameter")
        if not isinstance(dtype, str) or dtype not in supported:
            raise ValueError(f"{name}: unsupported restored parameter dtype {dtype}")
        parameter = parameters[name]
        if (
            not parameter.is_floating_point()
            or parameter.layout != torch.strided
            or parameter.device.type == "meta"
        ):
            raise ValueError(f"{name}: dtype restoration requires materialized dense floats")
        key = id(parameter)
        if key in planned and planned[key][1] != supported[dtype]:
            raise ValueError(f"{name}: tied parameters have conflicting restored dtypes")
        planned[key] = (parameter, supported[dtype])
    # Complete allocations before changing any Parameter. A failed conversion
    # leaves the model's precision and parameter ties intact.
    converted = [(parameter, parameter.detach().to(dtype)) for parameter, dtype in planned.values()]
    with torch.no_grad():
        for parameter, values in converted:
            parameter.data = values
