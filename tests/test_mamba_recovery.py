"""Exercise standalone state-space model recovery through weights and decoding."""

import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path

from rephix.cli import main
from rephix.weight_audit import DEFAULT_ATOL, DEFAULT_RTOL

HAS_HF = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "safetensors")
)


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class MambaRecoveryTests(unittest.TestCase):
    def _cli_json(self, args):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(args), 0)
        return json.loads(stdout.getvalue())

    def _recover(self, source, output, *, dtype="float64"):
        return self._cli_json(
            [
                "recover-layers",
                "--model",
                str(source),
                "--output",
                str(output),
                "--without-tokenizer",
                "--dtype",
                dtype,
                "--batch-size",
                "5",
            ]
        )

    def _check_export(self, config):
        import torch
        from transformers import AutoModelForCausalLM

        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "candidate"
            torch.manual_seed(413)
            target = AutoModelForCausalLM.from_config(config).eval()
            with torch.no_grad():
                for block in target.backbone.layers:
                    mixer = block.mixer
                    # Nondefault channel-specific dynamics and skips expose any
                    # attempt to leave the skeleton's parent parameters initialized.
                    mixer.A_log.uniform_(0.2, 1.8)
                    mixer.D.uniform_(-0.6, 0.9)
                    mixer.dt_proj.bias.fill_(-0.25)
                    if mixer.conv1d.bias is not None:
                        mixer.conv1d.bias.uniform_(-0.2, 0.2)
            target.double()
            for block in target.backbone.layers:
                # Native loaders construct A_log in FP32 independently of the
                # projection dtype used for high-precision probing.
                block.mixer.A_log.data = block.mixer.A_log.detach().float()
            target.generation_config.min_new_tokens = 3
            target.generation_config.max_new_tokens = 3
            target.save_pretrained(source, safe_serialization=True)

            report = self._recover(source, output)
            self.assertEqual(report["model_type"], config.model_type)
            self.assertEqual(
                report["unique_parameter_elements_recovered"],
                sum(parameter.numel() for parameter in target.parameters()),
            )
            audit = self._cli_json(["audit-weights", str(source), str(output)])
            self.assertEqual(audit["coverage"], 1.0)
            self.assertEqual(audit["fraction_within_tolerance"], 1.0)
            self.assertFalse(audit["extra_tensors"])
            self.assertFalse(audit["dtype_mismatches"])

            source.rename(Path(tmp) / "offline-source")
            candidate = AutoModelForCausalLM.from_pretrained(
                output, local_files_only=True, torch_dtype=torch.float64
            ).eval()
            embeddings = candidate.get_input_embeddings().weight
            self.assertEqual(
                candidate.get_output_embeddings().weight is embeddings, config.tie_word_embeddings
            )
            for expected, recovered in zip(target.backbone.layers, candidate.backbone.layers):
                self.assertEqual(recovered.mixer.conv1d.groups, config.intermediate_size)
                self.assertEqual(recovered.mixer.conv1d.padding, (config.conv_kernel - 1,))
                for name in ("A_log", "D"):
                    torch.testing.assert_close(
                        getattr(recovered.mixer, name),
                        getattr(expected.mixer, name),
                        atol=1e-7,
                        rtol=1e-6,
                    )

            inputs = torch.tensor([[4, 7, 5, 8, 9], [7, 6, 2, 10, 5]])
            with torch.inference_mode():
                expected = target(inputs, use_cache=False).logits
                torch.testing.assert_close(
                    candidate(inputs, use_cache=False).logits, expected, atol=2e-7, rtol=1e-6
                )
                for model in (target, candidate):
                    prefill = model(inputs[:, :-1], use_cache=True)
                    self.assertIsNotNone(prefill.cache_params)
                    step = model(
                        inputs[:, -1:],
                        use_cache=True,
                        cache_params=prefill.cache_params,
                        cache_position=torch.tensor([inputs.shape[1] - 1]),
                    )
                    torch.testing.assert_close(step.logits, expected[:, -1:], atol=2e-7, rtol=1e-6)
                generation = dict(use_cache=True, do_sample=False)
                expected_tokens = target.generate(inputs, **generation)
                self.assertEqual(expected_tokens.shape, (2, 8))
                self.assertTrue(
                    torch.equal(candidate.generate(inputs, **generation), expected_tokens)
                )

    def _configs(self):
        from transformers import FalconMambaConfig, MambaConfig

        common = dict(
            vocab_size=29,
            hidden_size=8,
            state_size=5,
            num_hidden_layers=2,
            expand=2,
            conv_kernel=3,
            time_step_rank=2,
            use_bias=True,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        return (
            MambaConfig(**common, tie_word_embeddings=True),
            FalconMambaConfig(**common, tie_word_embeddings=False, use_conv_bias=False),
        )

    def test_native_mamba_and_falcon_mamba_recover_weights_and_cached_generation(self):
        for config in self._configs():
            with self.subTest(model=config.model_type):
                self._check_export(config)

    def test_native_recurrence_cannot_distinguish_raw_logs_with_same_fp32_decay(self):
        import torch
        from transformers import AutoModelForCausalLM

        def first_log(model):
            return model.backbone.layers[0].mixer.A_log[0, 0]

        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            source, output = Path(tmp) / "source", Path(tmp) / "candidate"
            torch.manual_seed(351)
            lower = AutoModelForCausalLM.from_config(self._configs()[0]).eval()
            upper = deepcopy(lower)
            logs = torch.tensor([-2e-8, 2e-8], dtype=torch.float32)
            first_log(lower).copy_(logs[0])
            first_log(upper).copy_(logs[1])
            self.assertTrue(torch.equal(logs.exp(), torch.ones_like(logs)))

            ids = torch.tensor([[4, 7, 5]])
            a, b = [model(ids, use_cache=True) for model in (lower, upper)]
            self.assertTrue(torch.equal(a.logits, b.logits))
            for a_state, b_state in zip(a.cache_params.ssm_states, b.cache_params.ssm_states):
                self.assertTrue(torch.equal(a_state, b_state))

            lower.save_pretrained(source, safe_serialization=True)
            report = self._recover(source, output, dtype="float32")
            recovered = AutoModelForCausalLM.from_pretrained(output, local_files_only=True)
            fitted = first_log(recovered).float()
            self.assertEqual(fitted.exp().item(), 1.0)
            # The two original values differ by more than both tolerance bands combined.
            matches = torch.isclose(fitted, logs, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL)
            self.assertFalse(matches.all())
            first_mixer = report["state_space_decays"][0]
            self.assertEqual(first_mixer["module"], "backbone.layers.0.mixer")
            self.assertGreater(first_mixer["unidentified_parameter_elements"], 0)
            self.assertLess(
                report["individually_identified_parameter_elements"],
                report["unique_parameter_elements_recovered"],
            )

    def test_cuda_finds_native_representatives_for_rounded_and_saturated_rates(self):
        import torch
        from transformers.models.mamba.modeling_mamba import MambaMixer

        from rephix.ssm_recovery import observe_ssm_mixer, recover_ssm_mixer

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA for backend rounding coverage")
        config, _ = self._configs()
        target = MambaMixer(config, layer_idx=0).eval().cuda()
        fitted = MambaMixer(config, layer_idx=0).eval()
        logs = target.A_log.new_tensor([0.0050438442, 0.0084089469, 0.0618635565, -120, 110])

        def read(steps, shape):
            values = observe_ssm_mixer(torch, target, steps).cpu()
            self.assertEqual(values.shape, shape)
            return values

        with torch.no_grad():
            target.A_log[0].copy_(logs)
            target.D.uniform_(-0.6, 0.9)
            pending = {"A_log": fitted.A_log, "D": fitted.D}
            report = recover_ssm_mixer(torch, fitted, pending, read, rate_device="cuda")
            self.assertTrue(
                torch.equal(fitted.A_log.cuda().float().exp(), target.A_log.float().exp())
            )
            self.assertTrue(torch.equal(fitted.D, target.D.cpu()))
        self.assertEqual(report["saturated_effective_rate_elements"], 2)


if __name__ == "__main__":
    unittest.main()
