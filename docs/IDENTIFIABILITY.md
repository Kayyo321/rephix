# What parameter comparisons can establish

Weight recovery needs a specified model, a specified kind of observation, and a way to account for parameters that the observations cannot distinguish. A low training loss alone establishes none of these.

## An affine layer with known features

For `logits = features @ weights + bias`, full logits determine the affine parameters when the augmented feature matrix has full column rank. Rephix reports that rank and its conditioning. Too few independent observations leave a null space: multiple parameter matrices fit the same evidence. Regularization selects a numerical answer but does not supply missing observations.

Probabilities add an ambiguity even with full rank. Adding the same affine function to every class logit leaves softmax unchanged. The probability-based affine solver therefore reports the class-centered representative, rather than claiming the original uncentered weights were recovered.

## A hidden layer can change coordinates without changing its function

The tiny networks in the symmetry experiment use:

```text
y = activation(x @ incoming + hidden_bias) @ outgoing + output_bias
```

Reordering hidden neurons and applying the same reordering to outgoing rows preserves this function. For ReLU, multiplying one neuron's incoming weights and bias by a positive number `s`, and dividing its outgoing weights by `s`, also preserves it because `ReLU(s * z) = s * ReLU(z)`. For tanh, changing both incoming and outgoing signs preserves the function; arbitrary positive scaling does not.

Run a controlled example without model downloads:

```bash
rephix symmetry-demo --seed 42 --width 8 --activation relu --output runs/symmetry.json
rephix symmetry-demo --seed 42 --width 8 --activation tanh
```

The experiment plants a permutation and supported scales or signs. It reports the initial tensor distance, the distance after alignment, and output errors on separate observed and held-out inputs. The default ReLU example has substantial initial tensor distance despite matching outputs; alignment returns its parameters to the reference gauge within floating-point precision. This demonstrates why raw tensor distance can misrepresent recovery even for a very small network.

The NumPy API is `align_hidden(reference_in, reference_bias, reference_out, candidate_in, candidate_bias, candidate_out, activation="relu")`. Arrays follow the shapes in the formula above. `result.permutation[i]` names the original candidate neuron matched to reference neuron `i`; its incoming weights and bias are multiplied by `result.scales[i]` and its outgoing row is divided by that scale. The returned tensors preserve the candidate function. The output bias is not part of this hidden-neuron comparison and must be compared separately.

For ReLU, matching compares normalized incoming vectors, including hidden bias, and compensating outgoing weights. A Hungarian assignment minimizes the sum of squared distances between these descriptors. For tanh, matching chooses the better of the two allowed signs for each pair before assignment. This is a defined single-layer comparison, not a general minimizer over network functions or all possible parameter equivalences.

## Ambiguities that remain

Zero incoming weights and zero hidden bias make a neuron identically inactive for ReLU and tanh. Its outgoing weights are then unobservable. The implementation keeps its scale at one, preserves its outgoing weights, and reports the degeneracy rather than fabricating a recovered value. A zero outgoing row also hides its neuron's incoming parameters. Duplicate or canceling neurons can introduce additional ambiguities beyond the supported matching transformations.

Finite observations can also hide different functions, rather than merely different coordinates of the same function. The demo includes two width-two ReLU networks that agree at inputs `-1`, `0`, and `1`, yet differ at input `3`. Matching a finite prompt set does not establish agreement on unobserved prompts.

These alignment routines apply only to equal-width networks with one hidden layer and the stated activation. They do not align Qwen or other transformers, solve neuron splitting or merging, or prove unique recovery from prompts and outputs. Keep direct checkpoint distance, this restricted symmetry-aware comparison, behavioral agreement, and identifiable component recovery as separate measurements.
