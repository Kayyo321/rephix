# Recover latent autoencoders

`recover-diffusion` reconstructs native Diffusers `AutoencoderKL` and `AutoencoderTiny` components from chosen inputs to their convolution, normalization, and affine layers. The export preserves the native `config.json` and `diffusion_pytorch_model.safetensors` layout, and independently reloads through the original Diffusers class.

Install the optional runtime with `pip install -e '.[diffusion]'`. It uses Diffusers 0.35.x with PyTorch and safetensors, which is compatible with the project's Transformers dependency stack. Point `--model` at the local component directory containing its configuration and native safetensors archive. Standard and indexed Diffusers safetensors layouts are accepted. Legacy two-dimensional attention names are retained, and legacy aliases are exported in one native file for the loader's name conversion.

```sh
rephix --threads 2 recover-diffusion --model /path/to/source --output /path/to/recovered \
  --device cpu --dtype float64 --restore-source-dtypes --batch-size 512 \
  --evidence-dir /path/to/evidence
rephix --threads 2 replay-diffusion --evidence-dir /path/to/evidence --output /path/to/replayed
```

The trials used FP64 chosen-layer observations, followed by restoration of original storage precision. The evaluator independently loaded source and candidate in FP32 and compared encoder statistics, latent inputs, decoder results, and the ordinary forward entry point. Original tensor names and shapes were audited directly. The older KL archive uses 16 legacy attention projection names. The export retains those names through metadata aliases, so raw agreement covers the original archive layout.

## Published checkpoint trials

The trials used CPU execution with PyTorch 2.10.0 and Diffusers 0.35.2. Both checkpoints provide native safetensors directly. Every original stored value matched exactly. [Checkpoint hashes, raw audits, output signatures, and replay measurements](../examples/results/vae-layer-recovery.json)

| Reference | Original stored values | Exact numeric agreement | Held-out images | Encoder results | Maximum output error |
| --- | ---: | ---: | ---: | --- | ---: |
| [Stable Diffusion VAE FT MSE](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/31f26fdeee1355a5c34592e401dd41e45d25a493) | 83,653,863 | 100% | 9 | All 3,968 means, 3,968 log variances, and 3,968 fixed-seed posterior samples exact | 0 |
| [Tiny AutoEncoder for Stable Diffusion](https://huggingface.co/madebyollin/taesd/tree/614f76814bbe30edbe2e627ace1c2234c81a2c0e) | 2,445,063 | 100% | 9 | All 3,968 encoded latent values exact | 0 |

The image panel includes four random 64-by-64 images, four random 80-by-96 images, and a public image resized to 128-by-128 with bicubic interpolation and normalized to `[-1, 1]`. Input and output hashes are recorded for every panel. Each decoded output field contains 190,464 pixel coordinates across the nine images. All coordinates matched for decoding encoded latents, decoding independently chosen latents, and native forward execution. KL also matched all decoded posterior-sample coordinates with a reset generator. Tiny's native forward exercises its latent byte-quantization round trip. [Pinned public image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/dd11cba61e91b894782c9ea63fbd89456b758082/coco_sample.png)

Independent source-versus-source controls also produced exact outputs for every field. The KL export was republished from its sealed evidence after correcting legacy-key publication, reusing all 576 observations and collecting zero new responses. With the source paths absent, both checkpoints replayed from saved evidence. KL replay consumed 576 responses and Tiny consumed 202. Each reproduced its native configuration and safetensors file byte for byte with zero new calls.

These paths require chosen inputs and instrumented access to native internal operators. The evaluator uses fixed inputs and posterior generators to measure reconstruction fidelity.

```sh
python examples/vae_recovery_experiment.py \
  --reference /path/to/source --candidate /path/to/recovered \
  --model-id "$repo" --revision "$revision" --image /path/to/coco_sample.png \
  --device cpu --dtype float32 --threads 2 --output /path/to/results.json
```
