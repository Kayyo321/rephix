# Recover a native DDPM denoiser

`recover-diffusion` reconstructs supported native Diffusers model components through the existing chosen-layer probes. The published DDPM trial covers a complete `UNet2DModel`, including its convolutional residual blocks, attention projections, GroupNorm values, and timestep embedding network. It exports the native `diffusion_pytorch_model.safetensors` layout and preserves the scheduler configuration.

Install the tested optional runtime with `pip install -e '.[diffusion]'`. The trials use Diffusers 0.35.2. Sources are local component directories with a native configuration and safetensors archive.

```sh
rephix --threads 2 recover-diffusion --model /path/to/source --output /path/to/recovered \
  --device cpu --dtype float64 --restore-source-dtypes --batch-size 512 \
  --evidence-dir /path/to/evidence
rephix replay-diffusion --evidence-dir /path/to/evidence --output /path/to/replayed
```

The original DDPM archive uses legacy attention names such as `query`, `key`, `value`, and `proj_attn`. Current Diffusers converts those names when loading the model. Rephix records the original aliases and uses them when publishing the recovered archive, so raw audits compare every original stored tensor by name and shape. Replay preserves the same aliases.

## Published checkpoint trial

The trial used CPU FP64 probes followed by restoration of the original FP32 storage precision. Independent FP32 model reloads compared denoising predictions and a native stochastic DDPM trajectory. The source release provides safetensors directly. [Original checkpoint hashes, input hashes, raw audit, and trajectory measurements](../examples/results/diffusion-denoiser-recovery.json)

| Reference | Original stored values | Exact agreement | Exact denoiser outputs | Exact trajectory states |
| --- | ---: | ---: | ---: | ---: |
| [DDPM CIFAR-10 32](https://huggingface.co/google/ddpm-cifar10-32/tree/267b167dc01f0e4e61923ea244e8b988f84deb80) | 35,746,307 | 100% | 12 / 12 | 12 / 12 |

The denoiser panel uses square, rectangular, and batched inputs at timesteps 0, 125, 500, and 999. All 52,224 output coordinates matched exactly. The trajectory independently loads each checkpoint's preserved `DDPMScheduler`, starts from the same seeded noise, and uses the same seeded generator for the scheduler's stochastic updates. All 36,864 state coordinates and all twelve intermediate noise predictions matched exactly. Both maximum errors were zero.

With the original source path absent, replay consumed all 548 saved observations and made zero new calls. Its weights, configuration, and scheduler configuration matched all three initial export files byte for byte. These paths require chosen inputs and outputs at native internal layers.

The synthetic tests cover `AutoencoderKL`, `AutoencoderTiny`, `UNet2DModel`, and `UNet2DConditionModel`, including restored storage dtypes, independent native inference, original legacy attention names, and source-free replay. An interrupted denoiser capture committed four observations before a forced exception. Resumption reused those four responses and completed the original probe plan.

```sh
python examples/diffusion_denoiser_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered \
  --model-id google/ddpm-cifar10-32 \
  --revision 267b167dc01f0e4e61923ea244e8b988f84deb80 \
  --device cpu --dtype float32 --threads 2 --output /path/to/results.json
```
