"""Recover the observable gain of native Gemma offset RMSNorm modules."""

from __future__ import annotations

import math

from .rms_probe import unit_float32_rms_amplitude

GEMMA_OFFSET_RMS_NORMS = {
    "transformers.models.gemma.modeling_gemma.GemmaRMSNorm",
    "transformers.models.gemma2.modeling_gemma2.Gemma2RMSNorm",
    "transformers.models.gemma3.modeling_gemma3.Gemma3RMSNorm",
}
GEMMA3_SCALED_EMBEDDING = "transformers.models.gemma3.modeling_gemma3.Gemma3TextScaledWordEmbedding"


def recover_gemma_rms(torch, module, read):
    """Store a representative of the observed float32 gain and count ambiguous weights."""
    weight = module.weight
    shape = (1, *weight.shape)
    amplitude = unit_float32_rms_amplitude(module.eps, weight.numel())
    inputs = torch.full(shape, amplitude, dtype=torch.float32, device=weight.device)
    gain = read(inputs, shape)[0]
    weight.copy_(gain.double() - 1)
    if not torch.equal(1 + weight.float(), gain):
        raise ValueError("Gemma RMSNorm response has no representable gain")
    # The rounded gain is monotone, so a second preimage must include a neighbor.
    lower = torch.nextafter(weight, torch.full_like(weight, -math.inf))
    upper = torch.nextafter(weight, torch.full_like(weight, math.inf))
    return int(((1 + lower.float() == gain) | (1 + upper.float() == gain)).sum())
