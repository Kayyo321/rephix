# Output projection structure from dense log probabilities

The [known-feature trial](COMPONENT_RECOVERY.md) estimated specific Qwen output weights with access to the final hidden activations. This experiment gives the estimator a matrix of next-token log probabilities in fixed output coordinates. It estimates the output projection's contrast subspace and its numerical dimension. The local collector computes log probabilities from the model's public forward outputs. A separate evaluator reads the original weights after the fit has been saved. Neither the estimator nor the evaluator records hidden activations.

## Observations and identifiability

For context $i$, token $j$, and a common reference token $r$, log odds satisfy

$$
\log p_i(j)-\log p_i(r) = h_i^\top(w_j-w_r)+(b_j-b_r).
$$

This identity assumes an affine output projection and one shared softmax normalization per observation. Centering these contrasts across contexts removes the constant bias. The right singular vectors of the centered matrix estimate the range of the selected contrast weights in token-coordinate space. The estimate reaches the full range only when the queried contexts excite all relevant hidden directions and the selected token contrasts span them. A cutoff of $10^{-6}$ of the largest singular value was fixed before fitting. The report gives a sensitivity curve because small directions and observation noise can cross that threshold. The method follows the output-subspace argument of Carlini et al. [ICML 2024 paper](https://proceedings.mlr.press/v235/carlini24a.html)

A common shift across token logits cancels. An invertible change of hidden coordinates preserves the subspace while changing individual projection entries. The recovered basis is an equivalent representation of the exposed output space. The rank is conditional evidence about the effective width of the final projection. Hidden directions that never affect these probabilities may exist, and distinct architectures can share the same output dimension.

The experiment uses dense scores for 1,153 fixed token coordinates, including the reference. It assumes the endpoint exposes stable token identities and scores for those same coordinates on every context. Many deployed APIs expose only top-k results with support that varies by context. The corresponding control measures how much of the dense evidence is actually present in a top-20 response and leaves unobserved entries missing. Greedy text supplies less information still.

## Design and falsification

Each replicate scores 1,728 stateless next-token requests in 54 local model batches, with zero generated continuation tokens. A deterministic prompt generator freezes 1,536 fit prompts across 12 templates and 192 test prompts across six separate templates before any forward pass. Family indices are local to each split's template list. The prompt families cover synthetic language, numerical, code, and protocol fragments in several languages. The fit uses only training scores, with a fixed rank threshold. The first 1,001 selected probabilities on each test prompt serve as anchors. The estimator uses them to reconstruct the log odds of the remaining 152 token coordinates. This structural test requires new anchor scores from the target.

The ground-truth comparison uses the range of the selected, reference-subtracted output weights, in the same coordinate order. Relative projection-operator error divides the Frobenius norm of the projector difference by the square root of the reference rank. Predeclared acceptance checks require an estimated rank equal to the evaluator-only width, relative projection error below $10^{-3}$, and anchor prediction RMSE below $10^{-3}$ nats on separate prompt templates. A 128-row fit must exceed 0.1 nats in held-out projection RMSE. The controls also round training log probabilities to 0.001 nats and measure natural top-20 support on the same requests. The rounding control tests whether the fixed numerical rank rule survives lower precision. The top-k control tests whether the observations meet the input contract.

## Measured results

Both replicates passed all four checks on the pinned checkpoint in FP32. Seeds 7115 and 9821 changed the generated prompt variants, reference token, and fixed-coordinate sample while retaining the split design and model. [Machine-readable reports](../examples/results/logit-subspace.json)

| Measurement | Seed 7115 | Seed 9821 |
| --- | ---: | ---: |
| Estimated dense rank / configured width | 896 / 896 | 896 / 896 |
| Dense relative projection-operator error | 0.000122 | 0.000132 |
| Dense held-out anchor RMSE, nats | 0.000156 | 0.000175 |
| Fit-mean-only RMSE on the same coordinates, nats | 2.929 | 3.079 |
| Argmax agreement among 152 predicted coordinates | 192 / 192 | 192 / 192 |
| 128-row rank / held-out projection RMSE, nats | 127 / 1.629 | 127 / 1.668 |
| Rank with training scores rounded to 0.001 nats | 1,151 | 1,152 |
| Total prompt tokens scored | 44,006 | 44,019 |
| CUDA capture wall time | 5.17 s | 5.06 s |
| CPU fitting wall time for three conditions | 0.85 s | 0.83 s |

Peak PyTorch allocation was 2.02 GiB on the RTX 5080. Capture and fitting peaked at 3.68 GiB and 0.20 GiB process RSS respectively. Each replicate requested about 1.99 million fixed scores. The reports include environment versions, commands, artifact hashes, threshold sensitivity and evaluator runtime.

Refitting seed 7115 from the same captured training file reproduced all three fit archives byte for byte. Appending bytes to a copy of the dense fit archive caused the evaluator to reject the artifact before loading the reference checkpoint.

The dense estimate found rank 896 at relative cutoffs $10^{-6}$ and $10^{-7}$. Tightening the cutoff to $10^{-8}$ admitted noise directions in both runs. Rounding to 0.001 nats overestimated rank even at $10^{-6}$. In seed 9821, the rounded estimate projected test rows back with almost zero residual because it spanned every measured coordinate. Its projector differed substantially from the reference and the anchor system was underdetermined. The top-20 support omitted the chosen reference on every query in both replicates and contained at most two of the 1,153 fixed coordinates on any query. These results depend on access to dense, precise probability observations. There is no original-weight error estimate for this method.

## Reproduce

Use the pinned Qwen2.5-0.5B-Instruct checkpoint and isolated environment from the [local baseline](LOCAL_RESULTS.md). Each stage creates or uses directories outside the checkout. The fit process receives a copied training evidence file. The evaluator verifies the model hash, split artifacts, and fit artifacts before loading test scores and comparing with the original projection.

```bash
BASE="$HOME/.local/share/rephix-local-20261106"
RUN="$HOME/.local/share/rephix-subspace-reproduction"
PY="$BASE/venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=2
mkdir -p "$RUN"
for seed in 7115 9821; do
  "$PY" examples/logit_subspace_experiment.py capture \
    --model "$BASE/qwen" --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
    --run "$RUN/seed-$seed" --device cuda --batch-size 32 --seed "$seed"
  cp "$RUN/seed-$seed/train.npz" "$RUN/fit-evidence-$seed.npz"
  "$PY" examples/logit_subspace_experiment.py fit \
    --observations "$RUN/fit-evidence-$seed.npz" --run "$RUN/fit-$seed"
  "$PY" examples/logit_subspace_experiment.py test \
    --model "$BASE/qwen" --fit-dir "$RUN/fit-$seed" --run "$RUN/seed-$seed"
done
```

The next experiment needs a separately specified observation contract if dense scores are unavailable. For example, a controlled endpoint that exposes authorized token-logit offsets could expose more fixed log odds, with a separately measured request budget, numerical error and missingness. A mixed backend or a nonlinear post-projection transform could change the inferred span. The current trial assumes one fixed backend and evaluates a single checkpoint.
