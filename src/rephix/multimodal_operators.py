"""Chosen observations for native vision-language operators."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

CLIP_VISION_EMBEDDINGS = "transformers.models.clip.modeling_clip.CLIPVisionEmbeddings"
_LEGACY_LLAVA_IMAGE_NEWLINE_MODELS = {
    "transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration",
    "transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoForConditionalGeneration",
    "transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionForConditionalGeneration",
}
LLAVA_IMAGE_NEWLINE_MODELS = _LEGACY_LLAVA_IMAGE_NEWLINE_MODELS | {
    "transformers.models.llava_next.modeling_llava_next.LlavaNextModel",
    "transformers.models.llava_next_video.modeling_llava_next_video.LlavaNextVideoModel",
    "transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionModel",
}


def validate_clip_class_embedding(module, pending):
    if pending.keys() != {"class_embedding"} or pending["class_embedding"].shape != (
        module.embed_dim,
    ):
        raise ValueError("CLIP vision embeddings require one class vector")


def recover_clip_class_embedding(module, pending, read):
    weight = pending["class_embedding"]
    pixels = weight.new_zeros(
        (1, module.patch_embedding.in_channels, module.image_size, module.image_size)
    )
    weight.copy_(read(pixels, (1, module.embed_dim), operation="clip_class")[0])


def observe_clip_class_embedding(torch, module, inputs):
    # The class token includes a learned positional offset. Suppress the
    # position lookup output to observe the class vector before addition.
    handle = module.position_embedding.register_forward_hook(
        lambda _layer, _args, values: torch.zeros_like(values)
    )
    try:
        return module(inputs)[:, 0].to(dtype=inputs.dtype)
    finally:
        handle.remove()


def validate_llava_image_newline(module, pending):
    if pending.keys() != {"image_newline"} or pending["image_newline"].shape != (
        module.config.text_config.hidden_size,
    ):
        raise ValueError("LLaVA image packing requires one newline vector")


def recover_llava_image_newline(module, pending, read):
    weight = pending["image_newline"]
    shape = (1, module.config.text_config.hidden_size)
    weight.copy_(read(weight.new_zeros(shape), shape, operation="image_newline")[0])


def _observe_legacy_llava_image_newline(torch, module, inputs):
    # Older wrappers supply the newline to the native packer from forward.
    # Stop after packing so no language or vision weights are evaluated.
    class PackedImage(Exception):
        pass

    pack = module.pack_image_features

    def observe_pack(*args, **kwargs):
        packed, _ = pack(*args, **kwargs)
        raise PackedImage(packed[-1:])

    vision = module.config.vision_config
    token_ids = torch.zeros((1, 1), dtype=torch.long, device=inputs.device)
    sizes = token_ids.new_full((1, 2), vision.image_size)
    pixels = inputs.new_zeros((1, vision.num_channels, 1, 1))
    with (
        patch.object(module, "get_input_embeddings", return_value=lambda _: inputs.unsqueeze(0)),
        patch.object(module, "get_image_features", return_value=[inputs.unsqueeze(0)]),
        patch.object(module, "pack_image_features", side_effect=observe_pack),
    ):
        try:
            module(input_ids=token_ids, pixel_values=pixels, image_sizes=sizes)
        except PackedImage as observed:
            return observed.args[0]
    raise ValueError("LLaVA forward did not pack the chosen image features")


def observe_llava_image_newline(torch, module, inputs):
    # Drive the native image packer with chosen projected patches. No vision
    # or language weights are needed to expose the appended final row.
    if (
        f"{type(module).__module__}.{type(module).__qualname__}"
        in _LEGACY_LLAVA_IMAGE_NEWLINE_MODELS
    ):
        return _observe_legacy_llava_image_newline(torch, module, inputs)
    config = module.config
    vision = config.vision_config
    size = min(config.image_grid_pinpoints, key=lambda shape: shape[0] * shape[1])
    native = import_module(type(module).__module__)
    crops = native.image_size_to_num_patches(size, config.image_grid_pinpoints, vision.image_size)
    patches = (vision.image_size // vision.patch_size) ** 2
    pixels = inputs.new_zeros((crops, vision.num_channels, 1, 1))
    sizes = torch.tensor([size], device=inputs.device)
    tower_output = SimpleNamespace(hidden_states=(pixels,))
    features = inputs.expand(crops, patches, -1)
    with (
        patch.object(module.vision_tower, "forward", return_value=tower_output),
        patch.object(module.multi_modal_projector, "forward", return_value=features),
    ):
        packed = module.get_image_features(
            pixels, sizes, vision_feature_layer=0, vision_feature_select_strategy="full"
        )
    return packed[0][-1:]
