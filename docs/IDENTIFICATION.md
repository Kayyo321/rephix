# Behavioral identification

Rephix can run reconstruction with an unidentified target. Its optional fingerprint profile compares an exposed assistant response with reference responses to the same predeclared probe. It consumes text and observed time segments. The identifier never receives target model IDs, versions, token IDs, logprobs, config, weights, or tool metadata. In the controlled experiment, evaluator-only ground truth and generation settings were stored in `evaluator-plan.json`, separate from opaque run IDs and responses in `observed.json`. Prompts did not ask models to identify themselves. The method gives self-reported names and incidental metadata no separate evidentiary weight.

The profile measures lexical response similarity using a shared-prefix score and character-trigram overlap. The contrastive statistic multiplies a family's similarity by its positive margin over the closest reference rival. One-sided empirical ranks compare the observed statistic with complete-panel calibration runs from each reference deployment. The identifier accepts a lineage only when it beats its rivals and its compatibility rank exceeds the predeclared threshold. Incomplete panels still report raw ranked similarities, but cannot accept a lineage. The report shows individual probes that supported or weakened each hypothesis. The optional probe selector prioritizes outstanding probes with dissimilar responses across plausible reference families.

Compatibility ranks rely on exchangeability with the calibration population. They do not estimate posterior probabilities or general family-identification accuracy. The method reports version, architecture, and deployment configuration as unknown. A behavioral lineage match cannot resolve those properties. Separate complete observed time segments with different accepted lineages yield `mixed/routed`. A changed system prompt, wrapper, or generation configuration can mimic this evidence. A replacement backend and per-request routing cannot be distinguished from this signal alone.

## Local measurements

The eight fixed probes span literal formatting, creative continuation, code, math, tabular text, French, and classification. A greedy panel enrolled each of Qwen2.5-0.5B-Instruct, SmolLM2-135M-Instruct, and GPT-2. Twelve full calibration panels per checkpoint used temperature 0.70, top-p 0.90, and seeds 4100–4111. The threshold was 0.20. Each prompt was capped at 24 new tokens. We used native chat rendering when exposed by the checkpoint tokenizer and raw text otherwise. That access belonged to the controlled generator, outside the identifier.

An initial absolute similarity method accepted 15 of 24 matched sampling panels, all correctly. Overlapping profiles made the Qwen greedy control ambiguous, and a Pythia panel incorrectly matched GPT-2. A contrastive reanalysis of those same development observations accepted 21 of 24, all correctly, and abstained on all nine Pythia panels. We then locked the contrastive profile before generating the fresh holdout. Its SHA-256 was `2621b5abce1a362a4d163e721e72d311dc4f93a87afa008e641b39bc492986a7`. Repeated enrollment and calibration produced the same profile.

| Fresh holdout cohort | Panels | Correct known lineage | Abstained | Wrong lineage |
| --- | ---: | ---: | ---: | ---: |
| Same checkpoint and sampling settings, seeds 8300–8307 | 24 | 23 | 1 | 0 |
| Greedy controls on the enrolled checkpoints | 3 | 3 | 0 | 0 |
| Higher temperature, seeds 8300–8302 | 9 | 0 | 9 | 0 |
| Changed system instructions | 3 | 2 | 1 | 0 |
| Changed wrapper and prompt rendering | 3 | 1 | 2 | 0 |
| New checkpoint variants | 12 | 3 | 8 | 1 |
| Unfamiliar GPT-Neo-125M | 9 | 0 | 8 | 1 |

The raw highest-ranked family matched all 24 same-checkpoint sampling panels. For the new variants it matched 7 of 12 panels before abstention. The greedy Qwen2-0.5B-Instruct panel incorrectly matched SmolLM2, and one sampled GPT-Neo panel incorrectly matched GPT-2. Development also tested the Qwen2.5 and SmolLM2 base checkpoints against their instruction-tuned references. Changing checkpoint and rendering together confounds the separate effects of fine-tuning and formatting. These are descriptive counts over repeated prompts and a few fixed deployments. The fresh holdout used different seeds, checkpoints, wrapper text, and system instructions, but reused the preregistered diagnostic probes because the method depends on exact prompt matching.

For a controlled shift, we combined separate Qwen and GPT-2 greedy panels under two predeclared time segments. The report returned `mixed/routed` with the two individual matches. A shift from Qwen to GPT-Neo returned `unknown` overall with a Qwen match in the first segment. Alternating Qwen and GPT-2 within one panel returned `unknown`. This comparison reused collected panels and did not require more target queries. The measured cases support a narrow fingerprint for some enrolled deployments. More representative enrollment across variants and explicit controls separating checkpoint, template, and sampling effects would be needed for a family-wide claim.

Each phase made 816 target queries, including 312 enrollment and calibration queries, with at most 19,584 generated tokens per phase. Collection took 20.07 s in development and 17.99 s in holdout, including loading seven models per phase. The holdout peaked at 1.94 GiB Torch-allocated VRAM and 5.31 GiB process RSS. Execution used FP32 on the RTX 5080, Python 3.13.5, PyTorch 2.10.0+cu128, Transformers 4.57.6, and four host threads. The machine-readable results contain per-checkpoint scores, incorrect identifications, exact checkpoint digests, and hashes for the metadata-free evidence. Each run also writes individual predictions. [Measured results](../examples/results/identity-local.json) · [Frozen profile](../examples/results/identity-profile.json)

## Reproduce

Use the isolated environment and CUDA dependencies documented in [Local Qwen results](LOCAL_RESULTS.md). The following command installs no models into the repository. The example records downloaded-file digests in each evaluator manifest.

```bash
work="$HOME/.local/share/rephix-identity-reproduction"
mkdir -p "$work/models"

python - <<'PY' | while IFS=$'\t' read -r name repo revision; do
import runpy
for name, source in runpy.run_path("examples/identify_local_models.py")["MODELS"].items():
    print(name, source["repo"], source["revision"], sep="\t")
PY
  hf download "$repo" --revision "$revision" --local-dir "$work/models/$name" \
    --include config.json --include generation_config.json --include tokenizer.json \
    --include tokenizer_config.json --include special_tokens_map.json \
    --include model.safetensors --include merges.txt --include vocab.json
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python examples/identify_local_models.py --models "$work/models" \
  --qwen "$work/models/qwen-instruct" --output "$work/development" \
  --device cuda --threads 4 --phase development
python examples/identify_local_models.py --models "$work/models" \
  --qwen "$work/models/qwen-instruct" --output "$work/holdout" \
  --device cuda --threads 4 --phase holdout \
  --locked-profile "$work/development/contrastive/profile.json"
```

Use `--device cpu` for both phases to reproduce the method on CPU. Sampling and numerical outputs can differ across backends. Pass the resulting profile to a session only with the exact probe text and IDs. A full probe panel must fit within each observed segment for calibrated decisions. Outside the profiled deployment conditions, retain the ranked evidence and treat apparent matches as hypotheses.
