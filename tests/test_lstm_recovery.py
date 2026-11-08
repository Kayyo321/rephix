"""Verify native LSTM branch observations preserve independent packed parameters."""

import importlib.util
import unittest

from rephix.lstm_recovery import observe_lstm, recover_lstm, validate_lstm


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs PyTorch")
class LstmRecoveryTests(unittest.TestCase):
    def test_packed_affine_branches_recover_layers_directions_and_projection(self):
        import torch

        for options in (
            {"num_layers": 2, "bidirectional": True, "batch_first": True},
            {"num_layers": 2, "bias": False, "proj_size": 2},
        ):
            with self.subTest(options=options), torch.no_grad():
                source = torch.nn.LSTM(3, 4, dtype=torch.float64, **options)
                target = torch.nn.LSTM(3, 4, dtype=torch.float64, **options)
                for parameter in source.parameters():
                    parameter.copy_(torch.randint(-16, 17, parameter.shape).double() / 64)
                saved = {name: value.clone() for name, value in source.state_dict().items()}
                pending = dict(target.named_parameters(recurse=False))
                validate_lstm(target, pending)
                calls = []

                def read(inputs, shape, operation):
                    output = observe_lstm(torch, source, inputs, operation)
                    self.assertEqual(output.shape, shape)
                    calls.append(operation)
                    return output

                recover_lstm(torch, target, pending, read, batch_size=2)
                self.assertGreater(len(calls), len(pending))
                for name, actual in target.state_dict().items():
                    torch.testing.assert_close(actual, saved[name], rtol=0, atol=0)
                for name, actual in source.state_dict().items():
                    torch.testing.assert_close(actual, saved[name], rtol=0, atol=0)
                inputs = torch.randn(2, 3, 3, dtype=torch.float64)
                left, right = source(inputs), target(inputs)
                torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
                for expected, actual in zip(left[1], right[1]):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_gate_observations_distinguish_biases_hidden_by_their_sum(self):
        import torch

        with torch.no_grad():
            left = torch.nn.LSTM(3, 4, dtype=torch.float64)
            for parameter in left.parameters():
                parameter.copy_(torch.randint(-16, 17, parameter.shape).double() / 64)
            right = torch.nn.LSTM(3, 4, dtype=torch.float64)
            right.load_state_dict(left.state_dict())
            right.bias_ih_l0.add_(0.25)
            right.bias_hh_l0.sub_(0.25)
            inputs = torch.zeros(1, 1, 3, dtype=torch.float64)
            torch.testing.assert_close(left(inputs)[0], right(inputs)[0], rtol=0, atol=0)
            chosen = torch.zeros(1, 3, dtype=torch.float64)
            a = observe_lstm(torch, left, chosen, "gate_bias/bias_ih_l0")
            b = observe_lstm(torch, right, chosen, "gate_bias/bias_ih_l0")
            self.assertFalse(torch.equal(a, b))
            torch.testing.assert_close(b - a, torch.full_like(a, 0.25), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
