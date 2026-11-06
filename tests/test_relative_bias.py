"""Verify the attention-only T5-family position-bias probe."""

import importlib.util
import unittest

from rephix.relative_bias import recover_relative_bias, relative_bias_spec

HAS_HF = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(HAS_HF, "needs optional Transformers dependencies")
class RelativeBiasTests(unittest.TestCase):
    def test_reachable_buckets_recover_and_long_distance_bias_agrees(self):
        import torch
        from transformers import MT5Config, T5Config, UMT5Config
        from transformers.models.mt5.modeling_mt5 import MT5Attention
        from transformers.models.t5.modeling_t5 import T5Attention
        from transformers.models.umt5.modeling_umt5 import UMT5Attention

        cases = (
            (T5Config, T5Attention, False, 32, 128, [16]),
            (T5Config, T5Attention, True, 32, 128, []),
            (MT5Config, MT5Attention, False, 32, 128, [16]),
            (MT5Config, MT5Attention, True, 32, 128, []),
            (UMT5Config, UMT5Attention, False, 32, 128, [16]),
            (UMT5Config, UMT5Attention, True, 32, 128, []),
            (T5Config, T5Attention, False, 9, 9, [4, 8]),
            (T5Config, T5Attention, True, 31, 16, list(range(16, 30))),
        )
        for config_cls, attention_cls, decoder, buckets, distance, unreachable in cases:
            with self.subTest(model=config_cls.model_type, decoder=decoder, buckets=buckets):
                config = config_cls(
                    is_decoder=decoder,
                    d_model=12,
                    d_kv=4,
                    num_heads=3,
                    relative_attention_num_buckets=buckets,
                    relative_attention_max_distance=distance,
                )
                target = attention_cls(config, has_relative_attention_bias=True, layer_idx=0)
                fitted = attention_cls(config, has_relative_attention_bias=True, layer_idx=0)
                target_table = target.relative_attention_bias.weight
                fitted_table = fitted.relative_attention_bias.weight
                calls = []

                def read(positions, shape):
                    calls.append(positions.tolist())
                    result = target.compute_bias(
                        len(positions), shape[-1], cache_position=positions
                    )
                    self.assertEqual(tuple(result.shape), shape)
                    return result

                with torch.no_grad():
                    target_table.copy_(
                        torch.arange(1, target_table.numel() + 1).reshape_as(target_table)
                    )
                    report = recover_relative_bias(torch, fitted, read, batch_size=1)
                    expected = target_table.clone()
                    expected[unreachable] = 0
                    self.assertTrue(torch.equal(fitted_table, expected))
                    self.assertEqual(report["unreachable_buckets"], unreachable)
                    self.assertEqual(report["affected_stored_elements"], len(unreachable) * 3)
                    self.assertEqual(calls, [[distance + 1]] if decoder else [[distance + 1], [0]])
                    positions = torch.tensor([0, distance, distance * 3])
                    kwargs = {"key_length": distance * 2 + 1, "cache_position": positions}
                    self.assertTrue(
                        torch.equal(
                            target.compute_bias(3, **kwargs), fitted.compute_bias(3, **kwargs)
                        )
                    )

    def test_probe_reaches_outer_bucket_after_rounding_at_max_distance(self):
        import torch
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5Attention

        config = T5Config(
            is_decoder=True,
            d_model=4,
            d_kv=4,
            num_heads=1,
            relative_attention_num_buckets=8320,
            relative_attention_max_distance=4161,
        )
        target = T5Attention(config, has_relative_attention_bias=True, layer_idx=0)
        fitted = T5Attention(config, has_relative_attention_bias=True, layer_idx=0)

        def read(positions, shape):
            return target.compute_bias(len(positions), shape[-1], cache_position=positions)

        with torch.no_grad():
            report = recover_relative_bias(torch, fitted, read, batch_size=2)
        self.assertNotIn(8319, report["unreachable_buckets"])
        self.assertTrue(
            torch.equal(
                target.relative_attention_bias.weight[-1], fitted.relative_attention_bias.weight[-1]
            )
        )

    def test_bucket_device_matches_oracle_backend(self):
        import torch
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5Attention

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA for backend rounding coverage")
        config = T5Config(
            is_decoder=True,
            d_model=8,
            d_kv=4,
            num_heads=2,
            relative_attention_max_distance=25,
        )
        target = T5Attention(config, has_relative_attention_bias=True, layer_idx=0).cuda()
        fitted = T5Attention(config, has_relative_attention_bias=True, layer_idx=0)

        def read(positions, shape):
            self.assertEqual(positions.device.type, "cpu")
            return target.compute_bias(
                len(positions), shape[-1], cache_position=positions.cuda()
            ).cpu()

        with torch.no_grad():
            report = recover_relative_bias(torch, fitted, read, batch_size=2, bucket_device="cuda")
        expected = target.relative_attention_bias.weight.detach().cpu()
        expected[report["unreachable_buckets"]] = 0
        self.assertTrue(torch.equal(fitted.relative_attention_bias.weight, expected))

    def test_rejects_degenerate_or_oversized_bucket_config(self):
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5Attention

        for distance, error in ((8, "invalid logarithmic bins"), (1 << 16, "supported probe size")):
            with self.subTest(distance=distance):
                attention = T5Attention(
                    T5Config(relative_attention_max_distance=distance),
                    has_relative_attention_bias=True,
                )
                with self.assertRaisesRegex(ValueError, error):
                    relative_bias_spec(attention)

    def test_inconsistent_observation_leaves_table_unchanged(self):
        import torch
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5Attention

        config = T5Config(d_model=12, d_kv=4, num_heads=3)
        target = T5Attention(config, has_relative_attention_bias=True)
        fitted = T5Attention(config, has_relative_attention_bias=True)
        initial = fitted.relative_attention_bias.weight.detach().clone()

        def read(positions, shape):
            values = target.compute_bias(len(positions), shape[-1], cache_position=positions)
            values[0, 0, 0, 0] += 1
            return values

        with (
            torch.no_grad(),
            self.assertRaisesRegex(ValueError, "disagree within an expected bucket"),
        ):
            recover_relative_bias(torch, fitted, read, batch_size=2)
        self.assertTrue(torch.equal(fitted.relative_attention_bias.weight, initial))


if __name__ == "__main__":
    unittest.main()
