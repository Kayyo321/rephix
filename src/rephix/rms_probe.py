"""Inputs that make the native float32 RMS normalization factor equal one."""

import math


def unit_float32_rms_amplitude(eps, width):
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("RMSNorm epsilon must be finite and non-negative")
    _, exponent = math.frexp(eps)
    # The square is an exact power of two, with epsilon less than half an ulp.
    power = max(5, (exponent + 25) // 2)
    if 2 * power + (width - 1).bit_length() >= 128:
        raise ValueError("RMSNorm epsilon is too large for an exact float32 probe")
    return 2.0**power
