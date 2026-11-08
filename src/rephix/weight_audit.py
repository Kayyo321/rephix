"""Compare the stored coordinates of local safetensors checkpoints."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import numpy as np

DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 1e-5
_CHUNK_ELEMENTS = 1 << 20
_INT64_DTYPES = {"I64", "U64"}
_WIDE_DTYPES = {"F64", *_INT64_DTYPES}
_CAST_WITH_TORCH = {"BF16", "F8_E4M3", "F8_E5M2", "F8_E8M0"}


@dataclass(frozen=True)
class _Tensor:
    shard: Path
    shape: tuple[int, ...]
    dtype: str

    @property
    def size(self):
        return math.prod(self.shape)


@contextmanager
def _open_shard(path):
    if not path.is_file():
        raise ValueError(f"Cannot read safetensors shard {path}: expected a regular file")
    try:
        from safetensors import SafetensorError, safe_open
    except ImportError as exc:
        raise RuntimeError('Install optional dependencies with: pip install -e ".[hf]"') from exc
    try:
        with safe_open(path, framework="pt", device="cpu") as shard:
            yield shard
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise RuntimeError('Install optional dependencies with: pip install -e ".[hf]"') from exc
    except (OSError, SafetensorError) as exc:
        raise ValueError(f"Cannot read safetensors shard {path}: {exc}") from exc


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate key {name!r}")
        result[name] = value
    return result


def _shards(path):
    if not path.is_dir():
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError(f"Expected a local safetensors file or checkpoint directory: {path}")
        return {path: None}

    stem = "model"
    if not any(
        (path / filename).exists() or (path / filename).is_symlink()
        for filename in ("model.safetensors", "model.safetensors.index.json")
    ):
        stem = "diffusion_pytorch_model"
    index = path / f"{stem}.safetensors.index.json"
    if not index.exists() and not index.is_symlink():
        single_file = path / f"{stem}.safetensors"
        if not single_file.is_file():
            raise ValueError(
                f"Expected model.safetensors or diffusion_pytorch_model.safetensors "
                f"with an optional shard index in {path}"
            )
        return {single_file: None}

    try:
        if not index.is_file():
            raise ValueError("expected a regular index file")
        metadata = json.loads(
            index.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
        weight_map = metadata.get("weight_map") if isinstance(metadata, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("expected a nonempty weight_map object")
        shards = defaultdict(set)
        for name, filename in weight_map.items():
            if not isinstance(filename, str) or "\\" in filename or "\x00" in filename:
                raise ValueError(f"invalid shard path for {name!r}: {filename!r}")
            relative = Path(filename)
            if (
                relative.is_absolute()
                or PureWindowsPath(filename).drive
                or ".." in relative.parts
                or relative.suffix != ".safetensors"
            ):
                raise ValueError(f"invalid shard path for {name!r}: {filename!r}")
            # HF cache snapshots commonly link shard filenames to local content blobs.
            shards[path / relative].add(name)
        return shards
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid safetensors index {index}: {exc}") from exc


def _inventory(path):
    entries = {}
    for shard_path, expected in _shards(path).items():
        with _open_shard(shard_path) as shard:
            names = set(shard.keys())
            if expected is not None and names != expected:
                missing = sorted(expected - names)
                unindexed = sorted(names - expected)
                raise ValueError(
                    f"Shard {shard_path} disagrees with the index. "
                    f"Missing tensors: {missing}. Unindexed tensors: {unindexed}."
                )
            for name in names:
                tensor = shard.get_slice(name)
                entries[name] = _Tensor(shard_path, tuple(tensor.get_shape()), tensor.get_dtype())
    return entries


def _selections(shape):
    """Yield contiguous slices, including when a single row exceeds the chunk size."""
    if not shape:
        yield ()
        return
    if 0 in shape:
        return
    axis = len(shape) - 1
    trailing = 1
    while axis > 0 and trailing * shape[axis] <= _CHUNK_ELEMENTS:
        trailing *= shape[axis]
        axis -= 1
    step = _CHUNK_ELEMENTS // trailing
    for prefix in np.ndindex(shape[:axis]):
        fixed_axes = tuple(slice(index, index + 1) for index in prefix)
        for start in range(0, shape[axis], step):
            yield (*fixed_axes, slice(start, min(start + step, shape[axis])))


def _read_chunk(tensor, selection, entry, compute_dtype, name):
    chunk = tensor[selection]
    if entry.dtype in _CAST_WITH_TORCH:
        chunk = chunk.double()
    values = chunk.numpy()
    if entry.dtype in _INT64_DTYPES and np.finfo(compute_dtype).nmant < 63:
        if int(values.min()) < -(1 << 53) or int(values.max()) > (1 << 53):
            raise ValueError(
                f"Platform cannot compare large integers exactly in {entry.shard} tensor {name!r}"
            )
    values = values.astype(compute_dtype, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {entry.shard} tensor {name!r}")
    return values


@dataclass
class _Norm:
    """Accumulate squared norms in units of the largest observed magnitude."""

    scale: float = 0.0
    squares: float = 0.0

    def add(self, magnitudes):
        scale = magnitudes.max()
        if scale == 0:
            return
        normalized = (magnitudes / scale).astype(np.float64, copy=False).ravel()
        squares = float(np.einsum("i,i->", normalized, normalized))
        if scale > self.scale:
            self.squares = squares + self.squares * float(self.scale / scale) ** 2
            self.scale = scale
        else:
            self.squares += squares * float(scale / self.scale) ** 2

    def relative_to(self, reference):
        if self.scale == 0:
            return 0.0
        if reference.scale == 0:
            return None
        scale, exponent = np.frexp(self.scale)
        ref_scale, ref_exponent = np.frexp(reference.scale)
        mantissa = float(scale / ref_scale) * math.sqrt(self.squares / reference.squares)
        try:
            return math.ldexp(mantissa, int(exponent) - int(ref_exponent))
        except OverflowError as exc:
            raise ValueError("Relative L2 exceeds the finite float64 range") from exc


def _tolerance(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def compare_checkpoints(reference, candidate, *, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL) -> dict:
    """Compare same-name, same-shape tensors from local safetensors checkpoints.

    Fractions use all stored reference elements. Raw relative L2 requires full
    name and shape coverage. The zero-filled version substitutes zero for each
    missing or shape-incompatible tensor. Both use the full reference norm and
    exclude candidate extras. A zero reference norm gives zero for zero error
    and None for nonzero error. Fractions are None for an empty reference.
    """
    atol, rtol = _tolerance(atol, "atol"), _tolerance(rtol, "rtol")
    reference, candidate = Path(reference), Path(candidate)
    ref_entries, cand_entries = _inventory(reference), _inventory(candidate)
    missing = sorted(ref_entries.keys() - cand_entries.keys())
    extra = sorted(cand_entries.keys() - ref_entries.keys())
    common = sorted(ref_entries.keys() & cand_entries.keys())
    shape_mismatches = {
        name: {
            "reference": list(ref_entries[name].shape),
            "candidate": list(cand_entries[name].shape),
        }
        for name in common
        if ref_entries[name].shape != cand_entries[name].shape
    }
    dtype_mismatches = {
        name: {"reference": ref_entries[name].dtype, "candidate": cand_entries[name].dtype}
        for name in common
        if ref_entries[name].dtype != cand_entries[name].dtype
    }

    pairs = defaultdict(list)
    for name, entry in sorted(ref_entries.items()):
        other = cand_entries.get(name)
        other_shard = other.shard if other is not None and name not in shape_mismatches else None
        pairs[entry.shard, other_shard].append(name)

    reference_norm, error_norm = _Norm(), _Norm()
    compared_elements = exactly_equal = within_tolerance = 0
    for (ref_path, cand_path), names in pairs.items():
        cand_context = _open_shard(cand_path) if cand_path is not None else nullcontext()
        with _open_shard(ref_path) as ref_file, cand_context as cand_file:
            for name in names:
                ref_entry = ref_entries[name]
                cand_entry = cand_entries[name] if cand_file is not None else None
                compute_dtype = (
                    np.longdouble
                    if ref_entry.dtype in _WIDE_DTYPES
                    or (cand_entry is not None and cand_entry.dtype in _WIDE_DTYPES)
                    else np.float64
                )
                ref_tensor = ref_file.get_slice(name)
                cand_tensor = cand_file.get_slice(name) if cand_file is not None else None
                if cand_tensor is not None:
                    compared_elements += ref_entry.size
                for selection in _selections(ref_entry.shape):
                    a = _read_chunk(ref_tensor, selection, ref_entry, compute_dtype, name)
                    abs_a = np.abs(a)
                    reference_norm.add(abs_a)
                    if cand_tensor is None:
                        error_norm.add(abs_a)
                        continue
                    b = _read_chunk(cand_tensor, selection, cand_entry, compute_dtype, name)
                    with np.errstate(over="ignore"):
                        error = np.abs(b - a)
                        threshold = atol + rtol * abs_a
                    if not np.isfinite(error).all():
                        raise ValueError(
                            f"Difference exceeds the platform's finite range in tensor {name!r}"
                        )
                    error_norm.add(error)
                    exactly_equal += int(np.count_nonzero(a == b))
                    within_tolerance += int(np.count_nonzero(error <= threshold))

    # Check the finite-value invariant even for stored candidate coordinates outside both scores.
    unpaired = defaultdict(list)
    for name in sorted([*extra, *shape_mismatches]):
        unpaired[cand_entries[name].shard].append(name)
    for path, names in unpaired.items():
        with _open_shard(path) as shard:
            for name in names:
                entry = cand_entries[name]
                tensor = shard.get_slice(name)
                compute_dtype = np.longdouble if entry.dtype in _WIDE_DTYPES else np.float64
                for selection in _selections(entry.shape):
                    _read_chunk(tensor, selection, entry, compute_dtype, name)

    reference_elements = sum(entry.size for entry in ref_entries.values())
    zero_filled_relative_l2 = error_norm.relative_to(reference_norm)
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "atol": atol,
        "rtol": rtol,
        "scope": "raw stored safetensors tensors, including buffers, compared by name and shape",
        "reference_tensors": len(ref_entries),
        "candidate_tensors": len(cand_entries),
        "compared_tensors": len(common) - len(shape_mismatches),
        "reference_elements": reference_elements,
        "candidate_elements": sum(entry.size for entry in cand_entries.values()),
        "compared_elements": compared_elements,
        "coverage": compared_elements / reference_elements if reference_elements else None,
        "exactly_equal_elements": exactly_equal,
        "fraction_exactly_equal": exactly_equal / reference_elements
        if reference_elements
        else None,
        "within_tolerance_elements": within_tolerance,
        "fraction_within_tolerance": within_tolerance / reference_elements
        if reference_elements
        else None,
        "relative_l2": None if missing or shape_mismatches else zero_filled_relative_l2,
        "zero_filled_relative_l2": zero_filled_relative_l2,
        "missing_tensors": missing,
        "extra_tensors": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "interpretation": (
            "Numeric equality and tolerance use all stored reference elements. Tolerance is "
            "abs(candidate - reference) <= atol + rtol * abs(reference). Raw relative L2 requires "
            "full reference name and shape coverage. The zero-filled score substitutes zeros for "
            "missing or shape-incompatible tensors. Candidate extras are outside the scores. "
            "Serialized names can differ when checkpoints save tied weights under different aliases. "
            "This audit compares stored values without resolving aliases or aligning symmetries."
        ),
    }
