"""Native dense-image task routing and DPT position observations."""

from unittest.mock import patch

DENSE_VISION_TYPES = {"segformer", "dpt"}
DPT_EMBEDDINGS = "transformers.models.dpt.modeling_dpt.DPTViTEmbeddings"


def dense_vision_model_wrapper(config):
    from transformers import (
        AutoModelForDepthEstimation,
        AutoModelForImageClassification,
        AutoModelForSemanticSegmentation,
    )

    wrappers = {
        "SegformerForImageClassification": AutoModelForImageClassification,
        "SegformerForSemanticSegmentation": AutoModelForSemanticSegmentation,
        "DPTForDepthEstimation": AutoModelForDepthEstimation,
        "DPTForSemanticSegmentation": AutoModelForSemanticSegmentation,
    }
    architectures = config.architectures or ()
    if len(architectures) != 1 or architectures[0] not in wrappers:
        raise ValueError("dense image checkpoint must declare its native task architecture")
    architecture = architectures[0]
    if not architecture.lower().startswith(config.model_type):
        raise ValueError("dense image task architecture disagrees with its model type")
    return wrappers[architecture]


def validate_dpt_embeddings(module, pending):
    if pending.keys() != {"cls_token", "position_embeddings"}:
        raise ValueError("unsupported DPT embedding parameters")
    patches = module.patch_embeddings
    if (
        patches.image_size[0] != patches.image_size[1]
        or module.cls_token.shape != (1, 1, module.config.hidden_size)
        or module.position_embeddings.shape
        != (1, patches.num_patches + 1, module.config.hidden_size)
    ):
        raise ValueError("unsupported DPT image embedding layout")


def recover_dpt_embeddings(torch, module, pending, read):
    pixels = module.cls_token.new_zeros(
        (1, module.patch_embeddings.num_channels, *module.patch_embeddings.image_size)
    )
    pending["cls_token"].copy_(read(pixels, tuple(module.cls_token.shape), operation="dpt_class"))
    pending["position_embeddings"].copy_(
        read(pixels, tuple(module.position_embeddings.shape), operation="dpt_positions")
    )


def observe_dpt_embeddings(torch, module, inputs, operation):
    if operation == "dpt_class":
        positions = module.position_embeddings.new_zeros(module.position_embeddings.shape)
        with patch.object(module, "_resize_pos_embed", return_value=positions):
            return module(inputs)[0][:, :1]
    if operation != "dpt_positions":
        raise ValueError("unknown DPT embedding observation")
    patches = module.position_embeddings.new_zeros(
        (1, module.patch_embeddings.num_patches, module.config.hidden_size)
    )
    token = torch.nn.Parameter(torch.zeros_like(module.cls_token))
    with (
        patch.object(module, "cls_token", token),
        patch.object(module.patch_embeddings, "forward", return_value=patches),
    ):
        return module(inputs)[0]
