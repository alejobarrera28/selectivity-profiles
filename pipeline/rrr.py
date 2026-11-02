"""The model stage: the reduced-rank regression (RRR) model, fitted to all sessions.

The model (RRR) predicts each neuron's activity y from the 8 variables X, with
time-varying coefficients built from a few time shapes shared by all neurons.
It is fitted (fit) in two ways:
  - N_SPLITS times on 70% of each session's trials, to measure each neuron's
    R² on the 30% it didn't see (cross_validated_r2). The profiles use it to
    keep the well-fit neurons.
  - once on all trials, for the coefficients (fit_all_trials) used in the profiles.
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---- Model
RANK = 5  # d, number of shared time shapes (Posani-Wang: d = 5 was best)
L2 = 75   # lambda, ridge penalty: fixed to the middle of the Posani-Wang grid (25, 75, 200)

# ---- Cross-validation
N_SPLITS = 3         # Posani-Wang Methods: the split-train-test process is repeated 3 times
TEST_FRACTION = 0.3  # Posani-Wang code (their Methods say "one-third")
SPLIT_SEED = 42      # Posani-Wang code: split i uses random seed 42 + i

# ---- Fitting: L-BFGS settings of the Posani-Wang code
MAX_ITER = 1000
TOLERANCE_GRAD, TOLERANCE_CHANGE = 1e-7, 1e-4
HISTORY_SIZE = 50


# ================================================================ model

class RRR(torch.nn.Module):
    """The RRR encoding model (Posani-Wang Methods, eq. 2-3; Posani-Wang's RRRGD_model).

    For each neuron n, variable v and time bin t:
        beta[n, v](t) = sum_i U[n, v, i] * V[i](t)          (eq. 3)
        y_hat[n](k, t) = sum_v beta[n, v](t) * x[v](k, t) + b[n](t)
    V (rank x time bins) is shared by all neurons and sessions; each session
    has its own U (neurons x variables x rank) and b (neurons x time bins).
    Since X and y are z-scored, beta[n, v](t) reads: at time t, when variable v
    is 1 std above usual, neuron n's activity is beta std above usual.
    """

    def __init__(self, data, trials, rank):
        """`data` maps each session id to its inputs (the inputs stage's files),
        `trials` to the trials fitted.
        """
        super().__init__()
        torch.manual_seed(0)
        _, n_bins, n_variables = next(iter(data.values()))["X"].shape
        scale = 1 / np.sqrt(n_bins * rank)  # small random start (Posani-Wang code)
        self.V = torch.nn.Parameter(torch.randn(rank, n_bins, dtype=torch.float64) * scale)
        self.U = torch.nn.ParameterDict()
        self.b = torch.nn.ParameterDict()
        for eid in trials:
            _, y = _session_tensors(data[eid], trials[eid])
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

def _session_tensors(session, trials):
    """(X, y) of one session's trials, in float64 (Posani-Wang code)."""
    return (torch.tensor(session["X"][trials], dtype=torch.float64),
            torch.tensor(session["y"][trials], dtype=torch.float64))


def _session_loss(model, eid, x, y, l2):
    """Squared prediction error plus ridge penalty on the betas, for one session.

    Posani-Wang Methods, eq. 7. The baseline b is not penalized (Posani-Wang code).
    """
    error = ((model.predict(eid, x) - y) ** 2).sum()
    penalty = l2 * (model.beta(eid) ** 2).sum()
    return error + penalty


def fit(model, data, trials, l2, name):
    """Fit the model to all sessions at once with L-BFGS (Posani-Wang settings).

    `data` maps each session id to its inputs (the inputs stage's files),
    `trials` to the trials fitted. Shows a progress bar labelled `name`.
    Returns the loss after every evaluation, to see how the fit converged.
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
        for eid in trials:
            # Gradient accumulation: one session at a time, so only one is in memory in float64.
            x, y = _session_tensors(data[eid], trials[eid])
            loss = _session_loss(model, eid, x, y, l2)
            loss.backward()
            total += loss.item()
        losses.append(total)
        progress.update()
        progress.set_postfix(loss=f"{total:.4e}")
        return torch.tensor(total)

    optimizer.step(closure)  # runs up to MAX_ITER iterations
    progress.close()
    return losses


def fit_all_trials(data):
    """Fit on all trials of every session, for the coefficients (beta).

    `data` maps each session id to its inputs (the inputs stage's files).
    Returns the fitted model and its losses.
    """
    all_trials = {eid: np.arange(len(d["X"])) for eid, d in data.items()}
    model = RRR(data, all_trials, RANK)
    losses = fit(model, data, all_trials, L2, name="final fit (all trials)")
    return model, losses


# ================================================================ cross-validation

def split_trials(x, variables, split):
    """Training and test trial indices of one session, for one split.

    The test trials are TEST_FRACTION of the trials, drawn so that every block x
    contrast combination is in both parts (Posani-Wang Methods). If a session has too
    few trials of some combination, balance by block only, then not at all
    (Posani-Wang code). Also used by the profiles' null model, which needs the same splits.
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


def r2_spike_counts(y, y_hat, y_mean, y_std):
    """R² of each neuron, on spike counts (Posani-Wang's compute_R2_RRRGD).

    y and y_hat are z-scored (trials, time bins, neurons); they are un-z-scored
    first, and each neuron's R² is clipped at 0. Also used by the profiles' null model.

    The two versions measure different things. On z-scored data, R² is the
    share of the trial-to-trial variation the model explains. On spike counts,
    the average response over time is added back to both data and prediction,
    so it also counts as explained: R² is the share of all variation, and is
    higher. Posani-Wang's goodness of fit, and the profiles' threshold on R² minus
    the average-response model's R², are defined on firing rates, so we use the
    spike-count version.
    """
    y, y_hat = y * y_std + y_mean, y_hat * y_std + y_mean
    y, y_hat = y.reshape(-1, y.shape[2]), y_hat.reshape(-1, y.shape[2])  # (trials x bins, neurons)
    r2 = 1 - ((y - y_hat) ** 2).sum(axis=0) / ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return np.clip(r2, 0, None)


def test_r2(model, eid, x, y, y_mean, y_std):
    """R² of each neuron on held-out trials (x, y), from the model's predictions."""
    with torch.no_grad():
        y_hat = model.predict(eid, x).numpy()
    return r2_spike_counts(y, y_hat, y_mean, y_std)


def cross_validated_r2(data, split):
    """Fit on one split's training trials; R² of each neuron on its test trials.

    `data` maps each session id to its inputs (the inputs stage's files).
    Returns {eid: R² per neuron} and the fit's losses.
    """
    train, test = {}, {}
    for eid, d in data.items():
        train[eid], test[eid] = split_trials(d["X"], d["variables"], split)
    model = RRR(data, train, RANK)
    losses = fit(model, data, train, L2, name=f"split {split + 1}/{N_SPLITS}")
    r2 ={eid: test_r2(model, eid, torch.tensor(d["X"][test[eid]], dtype=torch.float64),
                       d["y"][test[eid]], d["y_mean"], d["y_std"])
          for eid, d in data.items()}
    return r2, losses
