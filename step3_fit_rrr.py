"""Step 3: fit the reduced-rank regression (RRR) model to all sessions.

The model (RRR) predicts each neuron's activity y from the 8 variables X, with
time-varying coefficients built from a few time shapes shared by all neurons.
It is fitted (fit) in two ways:
  - N_SPLITS times on 70% of each session's trials, to measure each neuron's
    R² on the 30% it didn't see (test_r2). Step 4 uses it to keep the
    well-fit neurons.
  - once on all trials, for the coefficients (beta) used in Fig. 2c.
Each fit is saved as soon as it is done and skipped if its file already exists,
so an interrupted run resumes where it stopped.

Output: in outputs/rrr/, the per-neuron test R² of each split and the final
model; a check plot in outputs/checks/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---- Data location
PREPROCESSED_DIR = "outputs/preprocessed"  # step 2's output

# ---- Outputs
RRR_DIR = "outputs/rrr"        # per-split test R² and the final model (used by step 4)
CHECKS_DIR = "outputs/checks"  # check plot

# ---- Model
RANK = 5  # d, number of shared time shapes (paper: d = 5 was best)
L2 = 75   # lambda, ridge penalty: fixed to the middle of the authors' grid (25, 75, 200)

# ---- Cross-validation
N_SPLITS = 3         # paper Methods: the split-train-test process is repeated 3 times
TEST_FRACTION = 0.3  # authors' code (the paper says "one-third")
SPLIT_SEED = 42      # authors' code: split i uses random seed 42 + i

# ---- Fitting: L-BFGS settings of the authors' code
MAX_ITER = 1000
TOLERANCE_GRAD, TOLERANCE_CHANGE = 1e-7, 1e-4
HISTORY_SIZE = 50


# ================================================================ model

class RRR(torch.nn.Module):
    """The RRR encoding model (paper Methods, eq. 2-3; authors' RRRGD_model).

    For each neuron n, variable v and time bin t:
        beta[n, v](t) = sum_i U[n, v, i] * V[i](t)          (eq. 3)
        y_hat[n](k, t) = sum_v beta[n, v](t) * x[v](k, t) + b[n](t)
    V (rank x time bins) is shared by all neurons and sessions; each session
    has its own U (neurons x variables x rank) and b (neurons x time bins).
    """

    def __init__(self, sessions, rank):
        """`sessions` maps each session id to its (X, y)."""
        super().__init__()
        torch.manual_seed(0)
        _, n_bins, n_variables = next(iter(sessions.values()))[0].shape
        scale = 1 / np.sqrt(n_bins * rank)  # small random start (authors' code)
        self.V = torch.nn.Parameter(torch.randn(rank, n_bins, dtype=torch.float64) * scale)
        self.U = torch.nn.ParameterDict()
        self.b = torch.nn.ParameterDict()
        for eid, (x, y) in sessions.items():
            n_neurons = y.shape[2]
            self.U[eid] = torch.nn.Parameter(
                torch.randn(n_neurons, n_variables, rank, dtype=torch.float64) * scale)
            # Start at the average response (~0). .contiguous() stores the
            # transposed values as a plain copy, which L-BFGS requires.
            self.b[eid] = torch.nn.Parameter(y.mean(axis=0).T.contiguous())

    def beta(self, eid):
        """Time-varying coefficients of a session's neurons: (neurons, variables, time bins)."""
        return self.U[eid] @ self.V

    def predict(self, eid, x):
        """Predicted y (trials, time bins, neurons) from X (trials, time bins, variables)."""
        # Einstein Summation Convention: for each trial k, time bin t and neuron n, sum
        # over the variables v of x[k, t, v] * beta[n, v, t]. Then add the baseline.
        return torch.einsum("ktv,nvt->ktn", x, self.beta(eid)) + self.b[eid].T


# ================================================================ fitting

def _as_tensors(data, trials):
    """{eid: trial indices} -> {eid: (X, y)} of those trials, in float64 (authors' code)."""
    return {eid: (torch.tensor(d["X"][trials[eid]], dtype=torch.float64),
                  torch.tensor(d["y"][trials[eid]], dtype=torch.float64))
            for eid, d in data.items()}


def _session_loss(model, eid, x, y, l2):
    """Squared prediction error plus ridge penalty on the betas, for one session.

    Paper Methods, eq. 7. The baseline b is not penalized (authors' code).
    """
    error = ((model.predict(eid, x) - y) ** 2).sum()
    penalty = l2 * (model.beta(eid) ** 2).sum()
    return error + penalty


def fit(model, sessions, l2, name):
    """Fit the model to all sessions at once with L-BFGS (authors' settings).

    Shows a progress bar labelled `name`. Returns the loss after every
    evaluation, to see how the fit converged.
    """
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=MAX_ITER,
                                  tolerance_grad=TOLERANCE_GRAD,
                                  tolerance_change=TOLERANCE_CHANGE,
                                  history_size=HISTORY_SIZE)
    losses = []
    # Out of MAX_ITER evaluations; the fit usually stops much earlier, when it converges.
    progress = tqdm(total=MAX_ITER, desc=name)

    def closure():
        # L-BFGS calls this whenever it needs the loss and its gradient.
        optimizer.zero_grad()
        total = 0.0
        for eid, (x, y) in sessions.items():
            loss = _session_loss(model, eid, x, y, l2)
            # Gradient accumulation: one session at a time, so only one is in memory.
            loss.backward()
            total += loss.item()
        losses.append(total)
        progress.update()
        progress.set_postfix(loss=f"{total:.4e}")
        return torch.tensor(total)

    optimizer.step(closure)  # runs up to MAX_ITER iterations
    progress.close()
    return losses


# ================================================================ cross-validation

def _split_trials(x, variables, split):
    """Training and test trial indices of one session, for one split.

    The test trials are TEST_FRACTION of the trials, drawn so that every block x
    contrast combination is in both parts (paper Methods). If a session has too
    few trials of some combination, balance by block only, then not at all
    (authors' code).
    """
    first_bin = x[:, 0, :]  # task variables are constant within a trial
    block = first_bin[:, list(variables).index("block")]
    contrast = first_bin[:, list(variables).index("contrast")]
    trials = np.arange(len(x))
    for balance_by in ([block, contrast], [block]):
        labels = [str(row) for row in np.stack(balance_by, axis=1).round(6)]
        try:
            return train_test_split(trials, test_size=TEST_FRACTION,
                                    random_state=SPLIT_SEED + split, stratify=labels)
        except ValueError:  # too few trials of some combination
            continue
    return train_test_split(trials, test_size=TEST_FRACTION, random_state=SPLIT_SEED + split)


def test_r2(model, eid, x, y, y_mean, y_std):
    """R² of each neuron on held-out trials (authors' compute_R2_RRRGD).

    The data and the predictions are un-z-scored first, and R² is computed on
    the smoothed spike counts; each neuron's R² is clipped at 0.

    The two versions measure different things. On z-scored data, R² is the
    share of the trial-to-trial variation the model explains. On spike counts,
    the average response over time is added back to both data and prediction,
    so it also counts as explained: R² is the share of all variation, and is
    higher. The paper's goodness of fit, and step 4's threshold on R² minus the
    average-response model's R², are defined on firing rates, so we use the
    spike-count version.
    """
    with torch.no_grad():
        y_hat = model.predict(eid, x).numpy()
    y, y_hat = y * y_std + y_mean, y_hat * y_std + y_mean
    y, y_hat = y.reshape(-1, y.shape[2]), y_hat.reshape(-1, y.shape[2])  # (trials x bins, neurons)
    r2 = 1 - ((y - y_hat) ** 2).sum(axis=0) / ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return np.clip(r2, 0, None)


# ================================================================ check

def plot_check(final, splits, dt, neural_start, path):
    """The final fit's loss and time shapes V, and the distribution of the test R².

    `final` and `splits` are the saved .npz files of the final and split fits.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    losses = final["losses"]
    axes[0].plot(100 * losses / losses[0])
    axes[0].set(xlabel="evaluation", ylabel="loss (% of starting loss)",
                ylim=(100 * losses.min() / losses[0] - 0.5, 100.5), title="final fit")
    t = neural_start + dt * (np.arange(final["V"].shape[1]) + 0.5)  # bin centres
    for i, shape in enumerate(final["V"]):
        axes[1].plot(t, shape, label=f"V{i + 1}")
    axes[1].axvline(0, color="k", lw=0.5)
    axes[1].set(xlabel="time from stimulus onset (s)", title="learned time shapes V")
    axes[1].legend(fontsize=7)
    eids = [key for key in splits[0].files if key != "losses"]
    r2 = np.concatenate([np.mean([split[eid] for split in splits], axis=0) for eid in eids])
    axes[2].hist(r2, bins=50)
    axes[2].set(xlabel=f"test R² (mean of {len(splits)} splits)", ylabel="neurons",
                title=f"per-neuron test R² (median {np.median(r2):.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ================================================================ main

if __name__ == "__main__":
    # 1. Load every preprocessed session. They stay in float32 to save memory;
    #    each fit converts the trials it uses to float64, as in the authors' code.
    data = {path.stem: dict(np.load(path)) for path in sorted(Path(PREPROCESSED_DIR).glob("*.npz"))}
    first = next(iter(data.values()))
    variables = first["variables"]
    dt, neural_start = float(first["dt"]), float(first["neural_start"])  # time axis, set in step 1
    print(f"Sessions: {len(data)}, neurons: {sum(d['y'].shape[2] for d in data.values())}")
    Path(RRR_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKS_DIR).mkdir(parents=True, exist_ok=True)

    # 2. The split fits: fit on 70% of the trials, measure each neuron's R² on the other 30%.
    for split in range(N_SPLITS):
        path = f"{RRR_DIR}/split{split}.npz"
        if not Path(path).exists():  # otherwise it was done in an earlier run
            train, test = {}, {}
            for eid, d in data.items():
                train[eid], test[eid] = _split_trials(d["X"], variables, split)
            training = _as_tensors(data, train)
            model = RRR(training, RANK)
            losses = fit(model, training, L2, name=f"split {split + 1}/{N_SPLITS}")
            del training  # free the memory before the next fit
            r2 = {eid: test_r2(model, eid, torch.tensor(d["X"][test[eid]], dtype=torch.float64),
                               d["y"][test[eid]], d["y_mean"], d["y_std"])
                  for eid, d in data.items()}
            np.savez(path, **r2, losses=losses)  # one R² per neuron, for each session

    # 3. The final fit, on all trials: its coefficients go to Fig. 2c.
    path = f"{RRR_DIR}/all_trials.npz"
    if not Path(path).exists():
        all_trials = _as_tensors(data, {eid: np.arange(len(d["X"])) for eid, d in data.items()})
        model = RRR(all_trials, RANK)
        losses = fit(model, all_trials, L2, name="final fit (all trials)")
        np.savez(path, V=model.V.detach().numpy(), losses=losses,
                 **{f"U_{eid}": model.U[eid].detach().numpy() for eid in data},
                 **{f"b_{eid}": model.b[eid].detach().numpy() for eid in data})

    # 4. Check plot and summary.
    splits = [np.load(f"{RRR_DIR}/split{split}.npz") for split in range(N_SPLITS)]
    final = np.load(f"{RRR_DIR}/all_trials.npz")
    plot_check(final, splits, dt, neural_start, f"{CHECKS_DIR}/step3_check.png")
    r2 = np.concatenate([np.mean([s[eid] for s in splits], axis=0) for eid in data])
    print(f"Done. Per-neuron test R²: median {np.median(r2):.3f}, "
          f"above 0 for {(r2 > 0).mean():.0%} of {len(r2)} neurons.")
