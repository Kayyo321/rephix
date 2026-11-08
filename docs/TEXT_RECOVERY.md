# Text-only recovery: contract, guarantee and limits

Rephix 0.4 answers the question left open in 0.3, "can model parameters be recovered
from text alone?", in two parts:

1. **Universal recovery of original parameters is impossible.** For any procedure and
   any query budget, there are targets whose text is indistinguishable while their
   parameters differ. `rephix text-identifiability` executes these counterexamples.
2. **Within a declared class, recovery is possible and implemented.** For every target
   whose next-output law is softmax-affine in a *public* feature map, `rephix
   text-recover` recovers an explicitly defined canonical parameterization from
   sampled strings. It also reports a standard error and tests whether the target is
   really in the class.

The second result is not a weaker restatement of the first. It is the strongest result
the impossibility argument permits: identification up to the exact equivalence that the
observations cannot resolve, inside a class that a statistical test can reject.

## Observation contract

| Item | Contract |
| --- | --- |
| Received by recovery | The prompt strings it sends and the strings the target returns |
| Query | `{"prompt": str, "n": int}` → `n` independent single-step continuations sampled at the target's documented temperature |
| Forbidden | Logits, probabilities, log probabilities, target token IDs, hidden states, weights, gradients, logit-bias or other decoding controls |
| Permitted prior | The feature specification: either a public token list and context order (`ngram`), or a public backbone checkpoint and the public label strings of a private head (`hf-backbone`) |
| Output | Canonical parameters with standard errors, a status, a transcript and a standalone export (`text-model`) that runs without the target |
| Budget | `--contexts`, `--samples-per-context`, `--max-queries` (ceiling on sampled outputs) |

The oracle is a separate process speaking JSON lines (`rephix-text-oracle/1`). The
client rejects any reply containing a field other than a list of exactly `n` strings,
so a misconfigured target cannot leak a privileged signal into the estimator. Every
output must equal one declared class string; any other string is a contract violation,
not a silently dropped sample. Transcripts are append-only JSONL with per-record
SHA-256 checksums and a plan hash, so interrupted runs resume without requerying.

In the experiments, the target and the evaluator run in different processes. Recovery
runs under a Python audit hook that rejects every attempt to open the private target
directory and logs every file it opens. The reload check runs under the same hook.
Ground truth is read only after recovery exits.

## The declared class

Let `a(x) = [phi(x); 1]` be the public feature vector of input `x` and `V` the number
of declared output strings. The class is

```text
p(y | x) = softmax(Theta a(x))_y,      Theta ∈ R^{V × (d+1)}
```

* `ngram`: `phi(x)` concatenates one-hot vectors of the last `k` tokens. This is the
  whole model: an additive log-linear k-gram language model.
* `hf-backbone`: `phi(x)` is the public backbone's final post-norm hidden state at
  the last token. The private part is the head `[W b]` over the declared labels, as in
  a fine-tuned classifier or a label-restricted endpoint built on a public base.

### Identification theorem

Let `S` be the span of `{a(x)}` over all valid inputs and `C = I - 11ᵀ/V`. Suppose the
queried inputs' feature vectors span `S`. Then for parameters `Theta`, `Theta'`:

> `p_Theta(·|x) = p_Theta'(·|x)` for every valid `x` **iff**
> `C Theta P_S = C Theta' P_S`, where `P_S` is the orthogonal projector onto `S`.

*Proof.* Equal softmax outputs at `x` are equivalent to logits differing by a constant
vector: `(Theta − Theta') a(x) = c(x) 1`. Multiplying by `C` (which annihilates `1`)
gives `C (Theta − Theta') a(x) = 0` for all queried `x`. Because these `a(x)` span
`S`, `C (Theta − Theta') P_S = 0`. Conversely, if that holds, then for any valid `x`,
`a(x) ∈ S`, so `C (Theta − Theta') a(x) = 0`, i.e. the logits differ by
`(1/V) 1 1ᵀ (Theta − Theta') a(x)`, a multiple of `1`, and the softmax outputs agree. ∎

Therefore `Theta_c = C Theta P_S` is the *canonical parameterization*. It is the most
that any text-only method can recover for this class. Recovering it also determines
the output law on **every** valid input, not only the queried ones. The equivalence it
leaves open is exactly `Theta ↦ Theta + 1 uᵀ + N`, where `N P_S = 0`. For `ngram`,
`dim S = k(V−1)+1`, because each one-hot block sums to the bias coordinate. For a
LayerNorm backbone with bias, `S` can be a hyperplane. The export stores `Theta_c` and
an orthonormal basis of `S`.

### Estimation and its guarantee

The counts of each returned string per context are multinomial draws from the law
above. `text-recover` maximizes the multinomial-logit likelihood in the reduced
coordinates `z = Qᵀa`, using matrix-free Newton–conjugate-gradient with a vanishing
ridge (`1e-6`). Because the likelihood is invariant to `Theta + 1uᵀ`, the penalized
optimum is automatically class-centred. The design has full rank on `S`, so the
log-likelihood is strictly concave on the canonical coordinates. Standard GLM theory
then makes the MLE consistent and asymptotically normal with covariance equal to the
inverse Fisher information (McCullagh & Nelder, *Generalized Linear Models*, 2nd ed.,
ch. 4–6). Rephix reports those standard errors and their relative norm. The error
shrinks as `1/√N` in the number of sampled strings.

Implementation and assumptions:

| Assumption | Where it is enforced |
| --- | --- |
| Outputs are independent samples, not greedy | Status `deterministic_outputs_not_identifiable` if every context returned a single string |
| Output strings map injectively to classes | Declared class strings must be distinct; an unknown string aborts the run |
| Queried features span `S` | Numerical rank equals the theoretical `dim S` (`ngram`) and validation inputs must lie in the span |
| Target is in the class | Parametric-bootstrap deviance test against the per-context saturated multinomial; rejected at `p ≤ alpha` (default 0.01, 99 replicates; fewer than `1/alpha − 1` replicates is refused because the test could never reject) |
| Enough samples for the claimed precision | Relative standard error must be ≤ `--max-relative-standard-error` (default 0.1) |

Only when every check passes is the status `recovered_in_class`. A rejected class test
yields `misspecified_approximation`. The export still runs, but it is a best
in-class approximation, not recovery. Anything else is `insufficient_evidence`.

**What would falsify the claim:** an in-class target whose canonical error exceeds its
reported uncertainty far more often than the confidence level allows, or an
out-of-class target labelled `recovered_in_class`. The pre-registered protocol tests
both.

**Numerical precision:** the estimator works in float64. Sampled strings carry finite
information, so recovery is always *within a statistical tolerance*. Bit-exact recovery
of real-valued parameters from finitely many strings is impossible. The `hf-backbone`
class assumes recovery computes the same features as the target. Both used float32 CPU
inference in the experiments. A different device or dtype can change features slightly
and appears as a small bias.

## Why universal recovery is impossible

`rephix text-identifiability` runs four checks. Results below were produced with seed 0.

| Check | Construction | Observable difference | Parameter difference |
| --- | --- | --- | --- |
| Hidden-unit permutation | MLP language model and its permuted copy, all 36 contexts | max probability gap 2.2e-16 | relative distance 1.12 |
| Transformer neuron permutation | Two-layer Qwen2 with MLP neurons permuted | max logit gap 0, identical sampled tokens | relative distance 0.21 |
| Greedy decoding | `Theta` and `2.5 Theta`, all contexts | identical greedy outputs | output laws differ (mean TV 0.26) |
| Finite budget | Order-3 target that differs from an order-2 target on one unqueried context | identical laws on all 24 queried prompts | trigger context TV 0.57 |
| In-class control | Gauge-shifted parameters with the same law | max law gap 2.2e-16 | canonical gap 7.5e-16 |

1. *Symmetry.* Any family with a function-preserving reparameterization has this
   problem: permuting hidden units, rescaling around ReLU or normalization layers,
   and the RMSNorm sign symmetry measured in 0.3. Such a family gives the same output
   law for `Theta` and `g·Theta`, at every query budget. No procedure can return
   "the original" for both, so original parameters are unidentifiable. At best, they
   are identified up to the symmetry group.
2. *Greedy decoding.* Text produced by argmax depends only on the ordering of logits.
   It cannot identify even the output law, let alone parameters.
3. *Finite queries.* For an unrestricted class, any finite set of prompts leaves an
   input where a richer target can differ. Agreement on tested prompts never implies
   agreement everywhere unless the class is restricted, as in the theorem above.

## Measured results

See [text recovery results](TEXT_RECOVERY_RESULTS.md) and the pre-registered
[protocol](../examples/text_recovery_protocol.json).

## Commands

```bash
# Run the counterexamples (add --numpy-only without the hf extra).
rephix text-identifiability --output runs/identifiability.json

# Serve a local controlled target as a string-only oracle and recover it.
rephix text-recover --features features.json --target target.json --output runs/export \
  --samples-per-context 8000 --validation-contexts 20
rephix text-model logprobs --model runs/export --prompt "abcd"
rephix text-model generate --model runs/export --prompt "abcd" --steps 20
```

`features.json` is `{"kind": "ngram", "vocabulary": [...], "order": 2}` or
`{"kind": "hf-backbone", "model": "/public/backbone", "labels": [...]}`. `target.json`
is `{"argv": [...]}` naming any program that speaks the oracle protocol on
stdin/stdout. `rephix text-target-serve --config PRIVATE.json` is the reference
implementation. HF-backbone recovery also needs `--contexts-file` with public prompts,
one per line.
