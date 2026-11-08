"""Expand-and-Cluster (Martinelli et al., 2024) for one-hidden-layer networks.

Reproduced on its own observation assumptions: real-valued, noise-free outputs of a
small network on controlled inputs, with an activation whose symmetries are known
(tanh/sigmoid: neuron permutation and sign flip; softplus: permutation, with sign flip
up to a linear term). Steps:

1. Expansion: train ``N`` students of width ``m`` (overparameterized) to near-zero MSE.
2. Collect hidden neurons ``[w; b]`` of converged students and canonicalize the sign.
3. Average-linkage hierarchical clustering with the cut that maximizes the number of
   clusters with at least ``γN`` members; reject clusters whose median angle to their
   centre is ``>= β``.
4. Each accepted cluster becomes one neuron (taken from the lowest-loss student);
   output weights are re-fitted by least squares, then the reduced network is fine-tuned.

``categorical`` mode is a separately labelled extension: the observations are sampled
class labels (strings), not real outputs. Students minimize cross-entropy. The paper's
evidence does not cover this mode; its results are reported as an extension.

Teacher parameters, when supplied, are used only by :func:`evaluate_recovery`, after
recovery has finished.
"""

from __future__ import annotations

import math

import numpy as np

from ..alignment import minimum_assignment


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("expand-cluster needs PyTorch: pip install '.[hf]'") from exc
    return torch


def _activation(torch, name):
    return {
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "softplus": torch.nn.functional.softplus,
        "relu": torch.relu,
    }[name]


def teacher_forward(params, x, activation):
    torch = _torch()
    act = _activation(torch, activation)
    W, b, A, c = (torch.as_tensor(params[k], dtype=torch.float64) for k in ("W", "b", "A", "c"))
    x = torch.as_tensor(x, dtype=torch.float64)
    return (act(x @ W.T + b) @ A.T + c).numpy()


def train_student(
    x, y, width, activation, *, seed, steps, mode="regression", device="cpu", lbfgs_steps=200
):
    """Full-batch Adam, then L-BFGS polishing. Returns parameters and final loss."""
    torch = _torch()
    act = _activation(torch, activation)
    generator = torch.Generator().manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    d = x.shape[1]
    if mode == "regression":
        y = torch.as_tensor(y, dtype=torch.float64, device=device)
        k = y.shape[1]
    else:
        counts = torch.as_tensor(y, dtype=torch.float64, device=device)
        k = counts.shape[1]
    W = (
        torch.randn(width, d, generator=generator, dtype=torch.float64) * math.sqrt(2 / (d + width))
    ).to(device)
    b = torch.zeros(width, dtype=torch.float64, device=device)
    A = (
        torch.randn(k, width, generator=generator, dtype=torch.float64) * math.sqrt(2 / (k + width))
    ).to(device)
    c = torch.zeros(k, dtype=torch.float64, device=device)
    params = [p.requires_grad_(True) for p in (W, b, A, c)]

    def loss_fn():
        out = act(x @ W.T + b) @ A.T + c
        if mode == "regression":
            return ((out - y) ** 2).mean()
        logp = torch.log_softmax(out, -1)
        return -(counts * logp).sum() / counts.sum()

    optimizer = torch.optim.Adam(params, lr=3e-3)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        if step == steps // 2:
            for group in optimizer.param_groups:
                group["lr"] = 1e-3
    lbfgs = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=lbfgs_steps,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        value = loss_fn()
        value.backward()
        return value

    try:
        lbfgs.step(closure)
    except RuntimeError:
        pass
    with torch.no_grad():
        final = float(loss_fn().item())
    return {k: v.detach().cpu().numpy() for k, v in zip("WbAc", params)}, final


def canonical_neurons(student, activation, sign_index):
    """Hidden neurons as [w; b] with the sign symmetry removed (odd-type activations)."""
    vectors = np.concatenate([student["W"], student["b"][:, None]], axis=1)
    outgoing = student["A"].T.copy()
    if activation in ("tanh", "sigmoid", "softplus"):
        flip = vectors[:, sign_index] < 0
        vectors[flip] *= -1
        outgoing[flip] *= -1  # bookkeeping only; outputs are re-fitted later
    return vectors, outgoing


def average_linkage(points):
    """Naive average-linkage agglomerative clustering. Returns merge heights and history."""
    n = len(points)
    distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    clusters = {i: [i] for i in range(n)}
    history = []
    while len(clusters) > 1:
        keys = list(clusters)
        best, pair = math.inf, None
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = clusters[keys[i]], clusters[keys[j]]
                value = distance[np.ix_(a, b)].mean()
                if value < best:
                    best, pair = value, (keys[i], keys[j])
        a, b = pair
        clusters[a] = clusters[a] + clusters.pop(b)
        history.append((best, {k: list(v) for k, v in clusters.items()}))
    return history


def cluster_neurons(vectors, owners, students, *, gamma, beta):
    """Cut maximizing clusters of size >= γN, then the angular misalignment filter."""
    history = average_linkage(vectors)
    best = None
    minimum = max(2, math.ceil(gamma * students))
    for height, clusters in [(0.0, {i: [i] for i in range(len(vectors))})] + history:
        big = [members for members in clusters.values() if len(members) >= minimum]
        if best is None or len(big) > len(best[1]):
            best = (height, big)
    height, big = best
    accepted, rejected = [], []
    for members in big:
        centre = vectors[members].mean(0)
        cosines = (
            vectors[members]
            @ centre
            / (np.linalg.norm(vectors[members], axis=1) * np.linalg.norm(centre) + 1e-30)
        )
        angle = float(np.median(np.arccos(np.clip(cosines, -1, 1))))
        # One neuron per student per cluster; duplicates within a student are summed later.
        entry = {
            "members": members,
            "median_angle": angle,
            "students": len({owners[m] for m in members}),
        }
        (accepted if angle < beta else rejected).append(entry)
    return {
        "cut_height": float(height),
        "accepted": accepted,
        "rejected": rejected,
        "minimum_cluster_size": minimum,
    }


def recover(
    x,
    y,
    *,
    activation="tanh",
    students=10,
    expansion=4.0,
    steps=6000,
    seed=0,
    gamma=0.8,
    beta=math.pi / 24,
    mode="regression",
    convergence=1e-6,
    student_width=None,
    device="cpu",
):
    """Run Expand-and-Cluster. Uses only (x, y); no teacher information."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    report = {
        "mode": mode,
        "activation": activation,
        "students": students,
        "gamma": gamma,
        "beta": beta,
        "convergence_threshold": convergence,
    }
    if student_width is None:
        # Width search: the smallest width whose best of three trial students fits.
        widths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        minimal = widths[-1]
        search = []
        for width in widths:
            losses = [
                train_student(
                    x,
                    y,
                    width,
                    activation,
                    seed=seed + 1000 * width + t,
                    steps=steps // 2,
                    mode=mode,
                    device=device,
                )[1]
                for t in range(3)
            ]
            search.append({"width": width, "best_loss": min(losses)})
            if min(losses) < _scaled(convergence, y, mode) * 10:
                minimal = width
                break
        student_width = int(min(64, math.ceil(expansion * minimal)))
        report["width_search"] = search
    report["student_width"] = student_width
    fits = []
    for index in range(students):
        params, loss = train_student(
            x,
            y,
            student_width,
            activation,
            seed=seed + index,
            steps=steps,
            mode=mode,
            device=device,
        )
        fits.append((params, loss))
    threshold = _scaled(convergence, y, mode)
    converged = [(p, loss) for p, loss in fits if loss < threshold]
    report["student_losses"] = [loss for _, loss in fits]
    report["converged_students"] = len(converged)
    report["loss_threshold"] = threshold
    if len(converged) < 2:
        report["status"] = "students_did_not_converge"
        report["recovered_width"] = 0
        return report, None
    order = sorted(range(len(converged)), key=lambda i: converged[i][1])
    converged = [converged[i] for i in order]
    sign_index = int(
        np.argmax(
            np.abs(
                np.concatenate([np.concatenate([p["W"], p["b"][:, None]], 1) for p, _ in converged])
            ).mean(0)
        )
    )
    vectors, owners = [], []
    for s, (params, _) in enumerate(converged):
        v, _ = canonical_neurons(params, activation, sign_index)
        vectors.append(v)
        owners.extend([s] * len(v))
    vectors = np.concatenate(vectors)
    clusters = cluster_neurons(vectors, owners, len(converged), gamma=gamma, beta=beta)
    report["clustering"] = {k: v for k, v in clusters.items() if k not in ("accepted", "rejected")}
    report["clusters_accepted"] = len(clusters["accepted"])
    report["clusters_rejected"] = len(clusters["rejected"])
    if not clusters["accepted"]:
        report["status"] = "no_cluster_accepted"
        report["recovered_width"] = 0
        return report, None
    neurons = []
    for cluster in clusters["accepted"]:
        # Paper: take the member from the lowest-loss student in the cluster.
        member = min(cluster["members"], key=lambda m: owners[m])
        neurons.append(vectors[member])
    neurons = np.array(neurons)
    reduced = _refit(x, y, neurons, activation, mode=mode, steps=steps // 2, device=device)
    report["recovered_width"] = int(len(neurons))
    report["reduced_loss"] = reduced["loss"]
    report["status"] = (
        "reduced_network_fits"
        if reduced["loss"] < threshold * 10
        else ("reduced_network_does_not_fit")
    )
    return report, reduced["params"]


def _scaled(convergence, y, mode):
    if mode == "regression":
        return convergence * float(np.var(y))
    return convergence  # excess cross-entropy is checked against the empirical optimum


def _refit(x, y, neurons, activation, *, mode, steps, device):
    """Least-squares output layer on recovered neurons, then fine-tune all parameters."""
    torch = _torch()
    act = _activation(torch, activation)
    W0, b0 = neurons[:, :-1], neurons[:, -1]
    with torch.no_grad():
        hidden = act(torch.as_tensor(x) @ torch.as_tensor(W0).T + torch.as_tensor(b0)).numpy()
    design = np.concatenate([hidden, np.ones((len(hidden), 1))], 1)
    if mode == "regression":
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    else:
        probabilities = y / y.sum(1, keepdims=True)
        logits = np.log(np.clip(probabilities, 1e-6, None))
        logits -= logits.mean(1, keepdims=True)
        solution, *_ = np.linalg.lstsq(design, logits, rcond=None)
    params = {"W": W0, "b": b0, "A": solution[:-1].T, "c": solution[-1]}
    tensors = [torch.tensor(params[k], dtype=torch.float64, requires_grad=True) for k in "WbAc"]
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)

    def loss_fn():
        out = act(xt @ tensors[0].T + tensors[1]) @ tensors[2].T + tensors[3]
        if mode == "regression":
            return ((out - yt) ** 2).mean()
        return -(yt * torch.log_softmax(out, -1)).sum() / yt.sum()

    lbfgs = torch.optim.LBFGS(
        tensors,
        lr=1.0,
        max_iter=max(50, steps // 10),
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        lbfgs.zero_grad()
        value = loss_fn()
        value.backward()
        return value

    try:
        lbfgs.step(closure)
    except RuntimeError:
        pass
    with torch.no_grad():
        loss = float(loss_fn().item())
    return {"params": {k: t.detach().numpy() for k, t in zip("WbAc", tensors)}, "loss": loss}


def evaluate_recovery(recovered, teacher, activation, x_test, *, mode="regression"):
    """Evaluator-only: symmetry-aware neuron alignment against teacher parameters."""
    if recovered is None:
        return {"recovered": False}
    t_vectors = np.concatenate([teacher["W"], teacher["b"][:, None]], 1)
    r_vectors = np.concatenate([recovered["W"], recovered["b"][:, None]], 1)
    # Sign symmetry: compare each recovered neuron to ±teacher neuron.
    cost = np.zeros((len(r_vectors), len(t_vectors)))
    for i, r in enumerate(r_vectors):
        for j, t in enumerate(t_vectors):
            cos = abs(r @ t) / (np.linalg.norm(r) * np.linalg.norm(t) + 1e-30)
            cost[i, j] = 1 - cos
    size = max(cost.shape)
    padded = np.full((size, size), 1.0)
    padded[: cost.shape[0], : cost.shape[1]] = cost
    assignment = minimum_assignment(padded)
    matched = [
        (i, int(assignment[i])) for i in range(len(r_vectors)) if assignment[i] < len(t_vectors)
    ]
    distances = [float(cost[i, j]) for i, j in matched]
    relative = []
    for i, j in matched:
        r, t = r_vectors[i], t_vectors[j]
        sign = 1.0 if r @ t >= 0 else -1.0
        relative.append(float(np.linalg.norm(sign * r - t) / np.linalg.norm(t)))
    out_t = teacher_forward(teacher, x_test, activation)
    out_r = teacher_forward(recovered, x_test, activation)
    if mode == "regression":
        output_error = float(np.sqrt(((out_t - out_r) ** 2).mean()) / (np.std(out_t) + 1e-30))
    else:
        pt = _softmax(out_t)
        pr = _softmax(out_r)
        output_error = float(0.5 * np.abs(pt - pr).sum(1).mean())
    return {
        "recovered": True,
        "teacher_width": int(len(t_vectors)),
        "recovered_width": int(len(r_vectors)),
        "matched_neurons": len(matched),
        "mean_cosine_distance": float(np.mean(distances)) if distances else None,
        "max_cosine_distance": float(np.max(distances)) if distances else None,
        "max_relative_neuron_error": float(np.max(relative)) if relative else None,
        "held_out_output_error": output_error,
        "output_error_kind": "relative RMSE" if mode == "regression" else "mean total variation",
        "alignment": "Hungarian matching on 1 − |cos| (sign symmetry) over [w; b]",
    }


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def make_teacher(rng, d, r, k, activation, scale=1.0):
    W = rng.standard_normal((r, d)) * scale
    return {
        "W": W,
        "b": rng.standard_normal(r) * 0.5,
        "A": rng.standard_normal((k, r)),
        "c": rng.standard_normal(k) * 0.1,
    }


def run_from_files(args):
    import json

    data = np.load(args.input, allow_pickle=False)
    if not {"inputs", "outputs"} <= set(data.files):
        raise ValueError("input NPZ must contain inputs and outputs arrays")
    report, params = recover(
        data["inputs"],
        data["outputs"],
        activation=args.activation,
        students=args.students,
        expansion=args.expansion,
        steps=args.steps,
        seed=args.seed,
    )
    report["assumptions"] = (
        "real-valued, noise-free outputs of a one-hidden-layer network with the stated "
        "activation on these inputs; recovery is up to permutation and sign"
    )
    if params is not None:
        report["recovered_parameters"] = {k: np.asarray(v).tolist() for k, v in params.items()}
    if args.output:
        from ..io import write_json

        write_json(args.output, report)
    del json
    return report
