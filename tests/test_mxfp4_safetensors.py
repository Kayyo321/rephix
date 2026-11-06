"""Native GPT-OSS MXFP4 layout and independent Transformers reload tests."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rephix.mxfp4_safetensors import audit_mxfp4_safetensors, recover_mxfp4_safetensors

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class NativeMxfp4Tests(unittest.TestCase):
    def _fixture(self, source, decoy=None):
        import torch
        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        if "gpt_oss" not in CONFIG_MAPPING:
            self.skipTest("installed Transformers has no native GPT-OSS model")
        config = AutoConfig.for_model(
            "gpt_oss",
            num_hidden_layers=1,
            num_local_experts=2,
            num_experts_per_tok=1,
            vocab_size=37,
            hidden_size=32,
            intermediate_size=32,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_scaling=None,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(54)
            model = AutoModelForCausalLM.from_config(config).eval().bfloat16()
        weights = model.state_dict()
        weights["model.layers.0.mlp.router.bias"] = weights[
            "model.layers.0.mlp.router.bias"
        ].float()
        weights["model.layers.0.self_attn.sinks"] = weights["model.layers.0.self_attn.sinks"].half()
        lut = np.array(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(44)
        matrices = {}
        for name, tensor in list(weights.items()):
            if not name.endswith((".gate_up_proj", ".down_proj")):
                values = tensor.float().numpy()
                matrices[name] = values.reshape(-1, values.shape[-1])
                continue
            del weights[name]
            experts, width, outputs = tensor.shape
            shape = (experts, outputs, width // 32)
            codes = rng.integers(0, 16, size=(*shape, 32), dtype=np.uint8)
            codes[0, 0, 0] = 1
            scales = np.full(shape, 123, dtype=np.uint8)
            weights[name + "_blocks"] = torch.from_numpy(codes[..., ::2] | (codes[..., 1::2] << 4))
            weights[name + "_scales"] = torch.from_numpy(scales)
            values = np.ldexp(lut[codes], scales.astype(np.int16)[..., None] - 127)
            matrices[name] = values.reshape(-1, width)

        for directory in (source, decoy):
            if directory is None:
                continue
            directory.mkdir()
            config.quantization_config = {"quant_method": "mxfp4"}
            config.save_pretrained(directory)
            (directory / "tokenizer_config.json").write_text('{"model_max_length": 64}')
            weight_map = {}
            for index in (1, 2):
                # Paired scales and codes may be stored in different shards.
                filename = f"model-0000{index}-of-00002.safetensors"
                payload = {
                    name: value if directory == source else torch.zeros_like(value)
                    for name, value in weights.items()
                    if name.endswith("_scales") == (index == 2)
                }
                save_file(payload, directory / filename, metadata={"format": "pt"})
                weight_map.update(dict.fromkeys(payload, filename))
            (directory / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": weight_map,
                        "metadata": {
                            "total_size": sum(
                                x.numel() * x.element_size() for x in weights.values()
                            )
                        },
                    }
                )
            )
        return matrices

    def test_external_observations_recover_from_payload_decoy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, decoy, candidate = (root / name for name in ("original", "decoy", "output"))
            matrices = self._fixture(reference, decoy)

            def oracle(name, start, stop, inputs):
                return inputs @ matrices[name][start:stop].T

            report = recover_mxfp4_safetensors(decoy, candidate, batch_size=7, call_tensor=oracle)
            self.assertEqual(report["oracle_device"], "external")
            self.assertEqual(
                report["observed_output_scalars"], sum(x.size for x in matrices.values())
            )
            self.assertGreater(report["mxfp4"]["ambiguous_blocks"], 0)
            audit = audit_mxfp4_safetensors(reference, candidate)
            self.assertEqual(audit["dense_values"], audit["exact_dense_values"])
            self.assertEqual(audit["mxfp4_codes"], audit["exact_mxfp4_values"])
            self.assertLess(audit["exact_mxfp4_scales"], audit["mxfp4_scales"])
            self.assertLess(audit["exact_mxfp4_codes"], audit["mxfp4_codes"])
            self.assertEqual(
                (decoy / "tokenizer_config.json").read_bytes(),
                (candidate / "tokenizer_config.json").read_bytes(),
            )

    def test_local_observations_match_independent_transformers_inference(self):
        import torch
        import transformers
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as directory:
            source, candidate = (Path(directory) / name for name in ("source", "candidate"))
            self._fixture(source)
            from transformers.integrations.mxfp4 import convert_moe_packed_tensors

            report = recover_mxfp4_safetensors(source, candidate, batch_size=11)
            self.assertEqual(report["oracle_device"], "cpu")
            direct_load = importlib.util.find_spec("accelerate") is not None and hasattr(
                transformers, "Mxfp4Config"
            )
            models = []
            with patch("torch.cuda.is_available", return_value=False):
                for path in (source, candidate):
                    if direct_load:
                        models.append(
                            AutoModelForCausalLM.from_pretrained(
                                path,
                                local_files_only=True,
                                torch_dtype=torch.bfloat16,
                                quantization_config=transformers.Mxfp4Config(dequantize=True),
                            ).eval()
                        )
                        continue
                    config = AutoConfig.from_pretrained(path, local_files_only=True)
                    del config.quantization_config
                    model = AutoModelForCausalLM.from_config(config).eval().bfloat16()
                    weights = {}
                    for shard in path.glob("*.safetensors"):
                        weights.update(load_file(shard))
                    for name in list(weights):
                        if name.endswith("_blocks"):
                            base = name.removesuffix("_blocks")
                            weights[base] = convert_moe_packed_tensors(
                                weights.pop(name), weights.pop(base + "_scales")
                            )
                    model.load_state_dict(weights)
                    models.append(model)
            tokens = torch.tensor([[5, 7, 9, 13, 19]])
            with torch.inference_mode():
                left, right = (model(tokens).logits for model in models)
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_original_openai_layout_preserves_structural_headers(self):
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as directory:
            source, candidate = (Path(directory) / name for name in ("source", "candidate"))
            source.mkdir()
            config = {
                "num_hidden_layers": 1,
                "num_experts": 2,
                "hidden_size": 32,
                "intermediate_size": 32,
            }
            (source / "config.json").write_text(json.dumps(config))
            weights = {"embedding.weight": np.eye(32, dtype=np.float32)}
            for name, rows in (("mlp1_weight", 64), ("mlp2_weight", 32)):
                prefix = f"block.0.mlp.{name}"
                weights[prefix + ".blocks"] = np.full((2, rows, 1, 16), 0x71, np.uint8)
                weights[prefix + ".scales"] = np.full((2, rows, 1), 123, np.uint8)
            filenames = ("model-0.safetensors", "model-1.safetensors")
            for filename, want_scales in zip(filenames, (False, True), strict=True):
                save_file(
                    {k: v for k, v in weights.items() if k.endswith(".scales") == want_scales},
                    source / filename,
                )
            report = recover_mxfp4_safetensors(source, candidate)
            audit = audit_mxfp4_safetensors(source, candidate)
            self.assertEqual(report["checkpoint_layout"], "openai_original")
            self.assertEqual(audit["exact_mxfp4_blocks"], 192)
            for filename in filenames:
                self.assertEqual(
                    (source / filename).read_bytes(), (candidate / filename).read_bytes()
                )


if __name__ == "__main__":
    unittest.main()
