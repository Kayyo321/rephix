# Create Mode: method specification

Status: written before the Create Mode implementation (Rephix 0.4.1). Later revisions
are recorded at the end of this file. Measured outcomes are in
[CREATE_MODE_RESULTS.md](CREATE_MODE_RESULTS.md); user-facing usage is in
[CREATE_MODE.md](CREATE_MODE.md).

Create Mode builds a new checkpoint inside a user-selected open-weight base model from
evidence about an unknown model or from a corpus of ordinary writing. This document
says precisely what is fitted, what is inferred, what can be identified, and what would
show the method failed. "Build Mode" is the same feature.

## 1. Inputs and notation

| Symbol | Meaning | Who supplies it |
| --- | --- | --- |
| `A` | Base architecture and tokenizer: config, tensor schema, special tokens, chat template, tying | User (local directory or pinned hub revision) |
| `θ_A` | Base weights | Loaded from `A` |
| `Θ_A` | All parameter values with `A`'s exact tensor names, shapes, dtypes and ties | Defined by `A` |
| `D` | Evidence: exchanges (ordered role messages, visible reasoning, tool calls, tool results) or passages (output-only text, corpus documents) | User |
| `C` | Optional instruction describing the desired creation | User, optional |
| `S` | The unknown source that produced `D` (absent for a corpus) | Never accessible to reconstruction |

Three different strings are kept apart in schema and code:

* the **observed prompt** (context that actually preceded an observed output, stored in
  the record),
* the **instruction** `C` (what the user wants created; stored in run provenance only),
* a **generation-time prompt** (what someone types when using the exported model).

`C` is never inserted into evidence records as a fabricated historical prompt.

## 2. Observables and capability tiers

Algorithm selection depends on which observables are actually present. The evidence
manifest records each tier as present or absent; nothing is inferred about the source to
fill a missing tier.

| Tier | Observable | Enables |
| --- | --- | --- |
| `corpus` | Output-only text with document boundaries, no prompts | Document-continuation fitting, style hypotheses |
| `passive_exchange` | Known context and response text | Prompt-conditional fitting and hypotheses |
| `structured_agent` | Tool calls, tool arguments, tool results, visible reasoning text | Tool behavior constraints and evaluation |
| `repeated_samples` | Several responses to an identical context | Information about the sampling law; required by the declared-class route |
| `source_token_probabilities` | Log probabilities attached to source tokens | Soft-target baselines only after a validated token mapping into `A` |
| `queryable_source` | An explicitly configured, authorized endpoint | Bounded acquisition of repeats and contrasts |
| `internal_traces` | Source activations or layer inputs/outputs | Not used by Create Mode; existing layer-recovery commands |

Visible reasoning is generated text. It is scored and fitted as text; it is not an
activation dump and not evidence of the computation that produced the answer. Tool results
are environment observations: they enter context and are excluded from the training loss.
Supplied tool calls are parsed as data and are never executed.

## 3. Unknowns, priors and the fitting problem

Create Mode solves the constrained problem

```text
θ̂ ∈ argmin_{θ ∈ Θ_A ∩ M}  L_obs(D, f_θ) + λ_prior R(θ, θ_A) + λ_retain L_retain(θ, θ_A)
                           + λ_instr L_instr(θ; C, θ_A)
```

* `L_obs`: token negative log-likelihood of the observed outputs given their observed
  context, under the declared serialization (assistant/passage tokens only).
* `R(θ, θ_A) = ||θ − θ_A||²` on the edited coordinates (the prior is "stay at the base").
* `L_retain`: KL(`p_θA` ‖ `p_θ`) on retain contexts that are unrelated to `D` (the base's
  own continuations of neutral prompts). It measures and limits collateral change.
* `L_instr` (only when `C` is given): KL(`p_θA(· | C, x)` ‖ `p_θ(· | x)`) on evidence
  contexts `x`. This is context distillation from the base itself. It moves the effect
  of `C` into weights, so the export needs no permanent prompt.
* `M` is the **structural constraint set**: `θ − θ_A` is nonzero only in the selected
  regions `r ∈ R̂`, with rank at most `k` per region, initialized in the inferred
  subspaces. `M` is chosen by the structural inference of section 5 before fitting.

This is a fitting problem. The regularizers select a solution among many that explain
`D`; they do not supply missing observations, and calling the problem "inverse" does not
establish recovery of `S`'s parameters. When `D` is the source's own sampled text,
minimizing `L_obs` is sequence-level distillation (Kim & Rush, 2016). With soft targets it
would be Hinton-style distillation; Create Mode uses soft targets only as a separately
labelled baseline when probabilities with a validated token mapping exist.

## 4. What can be identified

1. **Arbitrary unknown transformer from text: not identifiable.** Different parameters
   implement the same function (hidden-unit permutations; normalization rescalings;
   architecture-specific attention reparameterizations), and different functions agree
   on any finite set of observed contexts. `rephix text-identifiability` executes
   counterexamples. Greedy text identifies only logit order; sampled tokens are single
   draws, not distributions; repeated identical outputs do not prove greedy decoding.
   Unknown prompts, wrappers, retrieval and tools add further unobserved structure.
2. **Different architecture or tokenizer.** When `S` and `A` differ in width, depth,
   tokenizer or computation, a literal neuron correspondence may not exist. Re-expressing
   behavior in `A` is synthesis. Checkpoint serialization is not a bridge between
   architectures. Source token IDs are never read as base token IDs without a validated
   mapping.
3. **Declared region class (implemented, testable, rejectable).** Hypothesis `H_r,k`:
   `S` has architecture `A` and weights `θ_A + Δ`, where `Δ` is nonzero only in one
   region `r` (a single linear map of one layer) with rank at most `k`. For a linear map
   `W` with inputs `x`, outputs change only through `Δx`, so at best `Δ P_X` is
   identifiable, where `P_X` projects onto the span of inputs reached on the evidence
   distribution. Inside `H`, `Δ P_X` is generically locally identifiable from the full
   output law; from finite samples it is estimated with error. Outside `H` (edits in
   several regions, a reparameterized copy of a region, a different architecture) the
   estimate refers to the declared class, not to `S`. A function-preserving
   reparameterization of `S` (for example, an MLP neuron permutation in the edited layer)
   produces identical evidence while putting the edit in different coordinates. No test on
   `D` can detect that, so any recovered `Δ P_X` is stated as conditional on `H`.
4. **Declared-class output head (existing 0.4 route).** For
   `p(y | x) = softmax(Θ a(x))_y` with `a(x) = [φ(x); 1]`, public feature map `φ` and
   public labels, the identifiable object is `C Θ P_S` (Rephix 0.4,
   [TEXT_RECOVERY.md](TEXT_RECOVERY.md)). Create Mode reuses this estimator only when the
   evidence has repeated samples over a closed label set and `φ` is the selected base's
   final hidden state. That is the assumption that `S` uses `A`'s backbone on the claimed
   domain, and finite observations cannot establish it. Identification concerns the
   sampled law: parameter claims also need a known sampling law. Unknown temperature,
   truncation, rejection or wrapping changes the identified object or destroys
   identifiability. Failing to reject the class test is not proof of membership. Rank on
   observed contexts does not by itself establish coverage of every reachable feature
   direction. Finite-data output is an estimate conditional on the declared class and
   sampling assumptions, with standard errors. Nonexistence of the MLE (separation, empty
   classes) and ridge bias are reported.
5. **Small numerical-output networks (Expand-and-Cluster).** Martinelli et al. (2024)
   recover hidden neurons of small MLP teachers from noise-free real-valued outputs on
   chosen inputs, up to permutation and activation sign or scale symmetries. They train
   several overparameterized students and cluster recurring hidden weight vectors. The
   observation assumption (real-valued outputs, controlled input distribution, a known
   activation family) does not transfer to strings from a chat transformer. Rephix
   reproduces the method on its own assumptions and separately tests a string-only
   extension with its own negative controls. Success on a tiny MLP validates that
   component only.

## 5. Algorithms

### Stage 1: evidence preparation

Parse records into the explicit schema (`exchange` or `passage`). Missing context is
marked `missing`, never filled. Preserve the raw bytes and their SHA-256. Normalize
Unicode for grouping only. Union records into leakage groups by conversation, document,
explicit group, exact normalized duplicates and near-duplicates (MinHash on word
5-grams, Jaccard ≥ 0.7). Assign whole groups to train, validation and test, then freeze
the assignment before any fitting or selection. A single-document corpus is split into
contiguous section blocks. Produce the capability manifest.

Serialization is declared, not guessed. Exchanges use the base's chat template when it
has one. Otherwise they use the `rephix-plain-dialogue/1` convention. Visible reasoning
is wrapped in `<think>…</think>`; tool calls are serialized as
`<tool_call>{json}</tool_call>` inside the assistant turn. Loss masks cover assistant
tokens only. Passages use `document-continuation/1`: a document start uses the base's BOS
convention, a document end appends EOS, and windows never cross split boundaries. Text
with missing context starts mid-stream without BOS. All synthesized training pairs are
labelled as synthesized.

### Stage 2: hypotheses and observable constraints

For every training group, compute measurable text statistics:

* **Style:** sentence and paragraph length, word length, lexical variety, punctuation
  rates, function-word profile, contractions, pronoun person, hedges.
* **Discourse:** connective rates.
* **Formatting:** lists, headings, code, emphasis.
* **Length.**
* **Tools:** call rate, tool names, argument keys, order.
* **Reasoning presence.**

Content (distinctive topic words, quotations, references) is computed separately and is
never counted as style.

Compare these statistics with the base model's own continuations of the same training
contexts. A hypothesis "the evidence differs from the base in feature f" is
**supported** when the bootstrap interval of the standardized difference excludes zero on
training groups **and** the difference has the same sign on validation groups. Labels such
as "formal voice" remain hypotheses until the evaluation measures them on held-out data.
When `C` is given, its terms map to feature families (for example "formal" maps to
register and "paragraph" to paragraph structure) and are reported as instruction targets.
Unmapped terms are reported as unmapped.

### Stage 3: base analysis and structural inference

The candidate regions are the linear maps `q,k,v,o,gate,up,down` of every layer.
Embeddings, the output head, norms and buffers are excluded, so tied parameters stay tied.

For each region, Kronecker-factored model curvature is estimated on retain contexts
plus training-evidence contexts (inputs only):

* `A_r = E[x xᵀ]` over region inputs;
* `B_r = E[g gᵀ]` over output gradients under labels sampled from the base, which gives
  the true Fisher rather than the empirical Fisher.

Factors are dense up to 2048 features and diagonal beyond.

The evidence gradient `G_r` is computed on training evidence. The whitened gradient

```text
Ĝ_r = (B_r + λI)^{-1/2} G_r (A_r + λI)^{-1/2}
```

has the following property: for a rank-`k` edit with a fixed second-order KL budget
(measured from the base on the curvature contexts), the largest first-order decrease of `L_obs` is proportional to
`(Σ_{i≤k} σ_i(Ĝ_r)²)^{1/2}`. That quantity is the region score `s_r`. Its top singular
vectors define the inferred input and output subspaces.

Structural inference then produces three testable outputs:

1. **Region ranking.** Prediction: a fixed-budget natural-gradient trial edit in a
   high-scoring region lowers *held-out validation* NLL more than the same trial edit in
   a matched random region. Across screened regions, measured gain correlates positively
   with `s_r`. The step is scaled to a predicted KL `ε = 0.02` and halved until the
   *measured* KL on retain and training-evidence contexts (never validation) is at most
   `2ε`. A random rank-`k` direction of equal whitened norm in the same region is the
   direction control.
2. **Subspaces.** The edit for region `r` is `Δ_r = P_r M_r Q_rᵀ`, where `P_r` and `Q_r`
   are initialized from the whitened singular vectors (mapped back by the inverse
   whitening) and `M_r = 0`. Prediction: at matched parameter count and optimizer
   budget, this beats random orthonormal initial subspaces in the same regions.
3. **Feature descriptions.** Each write-side direction (`o_proj` and `down_proj`
   outputs) is projected through the unembedding (a logit-lens reading). The tokens it
   promotes and suppresses are reported as a hypothesis label. This is a description of a
   base-model direction, not a source neuron.

Selected regions: the top `R` by score, at most two per layer. These choices determine
the construction. They are not inspected after an ordinary fine-tune.

### Stage 4: parameter identification (capability-gated)

* **Region-class test (`H_r,k`).** Fit each of the top `K` regions alone, and the
  unrestricted structured edit (all selected regions), with equal steps. Compare
  validation NLL per held-out group with a paired bootstrap.
  * The best single region **explains** the change if its excess NLL over the
    unrestricted fit is below a tolerance `τ` (upper 95% bound) and it beats the
    runner-up region (lower 95% bound > 0).
  * If the best single region is significantly worse than the unrestricted fit, `H` is
    **rejected**.
  * If regions cannot be separated, the result is **not identified**.
  * When the region is explained, it is refit with different seeds. Seed-to-seed
    relative variation of `Δ_r X` (outputs on held-out region inputs) is reported as the
    uncertainty. Large variation is reported as nonuniqueness.
* **Declared-class head.** Runs only with repeated samples over closed labels, as in
  section 4 item 4.
* **Expand-and-Cluster.** Runs only for numerical-output evidence (`rephix
  expand-cluster`). Not applicable to text evidence.

### Stage 5: construction in A

`θ_created = θ_A + Δ`, with `Δ` supported on the selected regions. The adapters are
merged into standalone tensors in the base dtype. The export is then verified:

* the tensor schema equals the base schema (names, shapes, dtypes);
* ties are preserved;
* tokenizer files are byte-identical to the base's;
* logits after a fresh-process reload, through the ordinary runtime, match the
  in-process merged model within a stated tolerance.

No quantization or tokenizer conversion is performed. If one is ever added, it gets its
own check.

### Stage 6: evaluation and claim selection

The following measurements are separate:

* teacher-forced held-out NLL (validation for selection, test once at the end);
* free-running generation from held-out contexts and new-topic prompts, with style
  distance over supported hypotheses, base-model fluency of the generations, repetition,
  and long-span overlap with training text (memorization);
* retention: KL to base on neutral prompts, and factual cloze accuracy;
* tool schema validity, tool selection, arguments, order and mock outcomes;
* visible reasoning similarity, kept apart from final-answer correctness;
* runtime and memory.

Claims are stored separately:

| Claim | Status rule |
| --- | --- |
| Behavioral reconstruction | `supported` if test NLL improves on the base with a group-bootstrap 95% interval above zero **and** style distance does not increase; otherwise `not_supported` or `inconclusive` |
| Mechanistic synthesis in the base | `supported` if restoring each selected region to `θ_A` individually removes, summed over regions, ≥ 50% of the validation NLL gain (restoring all of them removes it by construction) **and** trial edits in selected regions beat matched random regions (one-sided permutation p < 0.05); otherwise `not_supported` |
| Partial source-parameter recovery | Only from Stage 4 routes whose assumptions are declared; otherwise `not_established` with the reason. Never set from behavioral agreement |
| Full source-parameter recovery | `not_established` for every Create Mode run: no implemented observation contract identifies a whole unknown model |

Process completion (`run_status`) is a separate field. A completed run with an
inconclusive experiment is a valid result. A behavioral-fitting-only backend
(`--method lora` or `--method sft`) is labelled as such, and its report states that
structural selection and source recovery were not attempted.

## 6. Falsification criteria

The structural contribution is falsified, and reported as such, if any of the
following holds on the pre-registered benchmarks:

* selected regions do not beat matched random regions at equal budget;
* inferred subspaces do not beat random subspaces;
* restoring selected regions does not reverse the effect;
* region scores do not correlate with measured trial gains;
* on planted same-architecture sources, the planted region is not ranked above chance.

Overclaiming is falsified if an out-of-class source receives an "explained" region
status, or if any run reports source-parameter recovery from output similarity alone.

## 7. What transfers from the references

| Reference | Used for | Not transferred |
| --- | --- | --- |
| Hinton et al. 2015; Kim & Rush 2016 | Naming: fitting sampled text is sequence-level distillation; soft targets are distillation | Nothing is claimed to avoid distillation |
| Vlačić & Bölcskei 2021 | Function-level identifiability up to sign and permutation for tanh-class networks given *f on all inputs* | Finite samples, ReLU/GELU/SiLU transformers, attention |
| Martinelli et al. 2024 | Expand-and-Cluster reproduction on its assumptions (real-valued, noise-free outputs, small MLPs) | Chat transformers, string-only outputs (tested separately as an extension) |
| Carlini et al. 2024 | Partial (projection-layer) recovery requires logit or logprob access and query control | Passive prose gives no such access |
| Meng et al. 2022 | Causal precedent: localize, edit accessible weights, restore, compare controls | Style is not a single factual association; no source-circuit claim |
| Elhage et al. 2022 | Describe features and subspaces rather than single neurons | Toy-model results are not assumed for real models |
| Rephix 0.4 text recovery | Declared-class head route, its theorem and its limits | Its guarantee does not cover an arbitrary source |

## Revision log

* Initial version, written before implementation.
* Engineering revision before any registered benchmark run (pilot seeds ≥ 100 only):
  1. Curvature uses retain plus training-evidence contexts. With retain-only curvature,
     natural steps exploded along directions the retain text does not use: trial edits
     raised validation NLL by about 10 nats.
  2. Trial steps backtrack on measured KL.
  3. The mechanistic rule uses summed single-region restoration, because restoring
     every region is trivially complete.
  4. Single-document corpora are split into about forty contiguous blocks instead of
     twenty.
  5. Passages declare a `joiner`, the exact text between a known context and the
     passage. Corpus sections use a blank line; prompt continuations use none.
