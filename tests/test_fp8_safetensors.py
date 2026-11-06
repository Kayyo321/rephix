"""Exercise FP8 sharding, partial blocks, and independent model inference."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "safetensors", "transformers")
)


def _decode(torch, weights, scales, block_size):
    rows, columns = weights.shape
    grid = scales.double().repeat_interleave(block_size[0], dim=0)
    grid = grid.repeat_interleave(block_size[1], dim=1)
    return weights.double() * grid[:rows, :columns]


def _weights(torch, shape, block_size, *, partial_alias=False):
    rows, columns = shape
    block_rows, block_columns = block_size
    rng = np.random.default_rng(45)
    codes = rng.integers(1, 127, size=shape, dtype=np.uint8)
    codes |= rng.integers(0, 2, size=shape, dtype=np.uint8) << 7
    scale_rows = (rows + block_rows - 1) // block_rows
    scale_columns = (columns + block_columns - 1) // block_columns
    scales = torch.tensor(
        rng.uniform(1.1e-4, 1.9e-4, size=(scale_rows, scale_columns)), dtype=torch.float32
    )
    for row in range(0, rows, block_rows):
        for column in range(0, columns, block_columns):
            codes[row, column] = 126
            if column + 1 < columns:
                codes[row, column + 1] = 1
            elif row + 1 < rows:
                codes[row + 1, column] = 1
    if partial_alias:
        # An interior code in a singleton tail admits alternative scales.
        codes[-1, -1] = 48
    return torch.from_numpy(codes).view(torch.float8_e4m3fn), scales


def _qwen_checkpoint(path, block_size):
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, Qwen2Config

    config = Qwen2Config(
        num_hidden_layers=1,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=41,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng():
        torch.manual_seed(38)
        model = AutoModelForCausalLM.from_config(config).eval()
    tensors = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            codes, scales = _weights(torch, module.weight.shape, block_size)
            tensors[name + ".weight"] = codes
            # Both FP32 and BF16 scales are found in published checkpoints.
            if name.endswith(".k_proj"):
                scales = scales.to(torch.bfloat16)
            tensors[name + ".weight_scale_inv"] = scales
    path.mkdir()
    config.quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": list(block_size),
    }
    config.save_pretrained(path)
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    del config.quantization_config
    return config


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class Fp8SafetensorsTests(unittest.TestCase):
    def test_external_probes_ignore_decoy_and_preserve_sharded_partial_blocks(self):
        import torch
        from safetensors.torch import load_file, save_file

        from rephix.fp8_safetensors import audit_fp8_safetensors, recover_fp8_safetensors

        block_size = (3, 4)
        weight_name = "model.layers.0.mlp.up_proj.weight"
        scale_name = weight_name + "_scale_inv"
        dense_name = "model.embed_tokens.weight"
        codes, scales = _weights(torch, (7, 9), block_size, partial_alias=True)
        dense = torch.arange(20, dtype=torch.bfloat16).reshape(4, 5) / 8
        decoded = _decode(torch, codes, scales, block_size)
        tensors = {weight_name: codes, scale_name: scales, dense_name: dense}
        names = {
            weight_name: "model-00001-of-00002.safetensors",
            dense_name: "model-00001-of-00002.safetensors",
            scale_name: "model-00002-of-00002.safetensors",
        }
        config = {
            "model_type": "qwen2",
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": list(block_size),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, decoy = root / "source", root / "decoy"
            for folder in (source, decoy):
                folder.mkdir()
                (folder / "config.json").write_text(json.dumps(config))
                (folder / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": names})
                )
                for shard in set(names.values()):
                    payload = {
                        name: value if folder == source else torch.zeros_like(value)
                        for name, value in tensors.items()
                        if names[name] == shard
                    }
                    save_file(payload, folder / shard, metadata={"format": "pt"})
            for mode, dtype in (("float64", torch.float64), ("float32", torch.float32)):
                with self.subTest(mode=mode):
                    candidate = root / mode
                    observed = {
                        weight_name: decoded.to(dtype).numpy(),
                        dense_name: dense.to(dtype).numpy(),
                    }
                    calls = []

                    def oracle(name, start, stop, inputs):
                        self.assertEqual(inputs.dtype, np.dtype(mode))
                        calls.append(name)
                        return inputs @ observed[name][start:stop].T

                    recover_fp8_safetensors(
                        decoy, candidate, batch_size=3, call_tensor=oracle, observation_dtype=mode
                    )
                    self.assertEqual(set(calls), {weight_name, dense_name})
                    for filename in ("config.json", "model.safetensors.index.json"):
                        self.assertEqual(
                            (candidate / filename).read_bytes(), (source / filename).read_bytes()
                        )
                    recovered = {}
                    for shard in set(names.values()):
                        recovered.update(load_file(candidate / shard))
                    self.assertEqual(recovered[weight_name].dtype, torch.float8_e4m3fn)
                    actual = _decode(
                        torch, recovered[weight_name], recovered[scale_name], block_size
                    )
                    torch.testing.assert_close(actual.to(dtype), decoded.to(dtype), atol=0, rtol=0)
                    torch.testing.assert_close(recovered[dense_name], dense, atol=0, rtol=0)
                    audit = audit_fp8_safetensors(source, candidate)
                    self.assertEqual(audit["fp8_codes"], codes.numel())
                    self.assertEqual(audit["fp8_scales"], scales.numel())
                    if mode == "float64":
                        self.assertEqual(audit["exact_fp8_values"], codes.numel())
                    self.assertEqual(audit["exact_fp32_values"], codes.numel())
                    self.assertEqual(audit["exact_dense_values"], dense.numel())

    def test_recovered_qwen_matches_independent_torch_dequantized_inference(self):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM

        from rephix.fp8_safetensors import recover_fp8_safetensors

        block_size = (128, 128)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, candidate = root / "source", root / "candidate"
            config = _qwen_checkpoint(source, block_size)
            recover_fp8_safetensors(source, candidate, batch_size=41)

            tokens = torch.tensor([[1, 7, 19, 4, 12]])

            def inference(checkpoint):
                weights = load_file(checkpoint / "model.safetensors")
                for key in tuple(weights):
                    if key.endswith(".weight_scale_inv"):
                        name = key.removesuffix("_scale_inv")
                        weights[name] = _decode(
                            torch, weights[name], weights.pop(key), block_size
                        ).float()
                model = AutoModelForCausalLM.from_config(config).eval()
                model.load_state_dict(weights, strict=True)
                with torch.inference_mode():
                    return model(tokens, use_cache=False).logits

            torch.testing.assert_close(inference(source), inference(candidate), atol=0, rtol=0)

    @unittest.skipUnless(
        all(importlib.util.find_spec(name) is not None for name in ("accelerate", "triton")),
        "needs optional accelerate and triton for the native FP8 runtime",
    )
    def test_recovered_qwen_loads_with_native_finegrained_fp8(self):
        import torch
        import transformers

        from rephix.fp8_safetensors import recover_fp8_safetensors

        if not hasattr(transformers, "FineGrainedFP8Config"):
            self.skipTest("installed Transformers has no native fine-grained FP8 runtime")
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("needs CUDA compute capability 8.9 or newer")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, candidate = root / "source", root / "candidate"
            _qwen_checkpoint(source, (128, 128))
            recover_fp8_safetensors(source, candidate, batch_size=41)
            tokens = torch.tensor([[1, 7, 19, 4, 12]], device="cuda")

            def inference(path):
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    path,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                ).eval()
                self.assertEqual(
                    model.model.layers[0].self_attn.q_proj.weight.dtype, torch.float8_e4m3fn
                )
                with torch.inference_mode():
                    return model(tokens, use_cache=False).logits.cpu()

            torch.testing.assert_close(inference(source), inference(candidate), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
