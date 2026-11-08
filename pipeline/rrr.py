"""The model stage: the reduced-rank regression (RRR) model, fitted to all sessions.

The model (RRR) predicts each neuron's activity y from the 8 variables X, with
time-varying coefficients built from a few time shapes shared by all neurons.
It is fitted (fit) in two ways:
  - N_SPLITS times on 70% of each session's trials, to measure each neuron's
    R² on the 30% it didn't see (cross_validated_r2). The profiles use it to
    keep the well-fit neurons.
  - once on all trials, for the coefficients (fit_all_trials) used in the profiles.

The loss is the squared prediction error summed over trials, plus a ridge
penalty (Posani-Wang Methods, eq. 7). The optimizer evaluates it hundreds of
times per fit, once for each set of coefficients it tries. The Posani-Wang code
predicts every trial in every evaluation. Here, expanding the square shows that
the error depends on a session's trials only through three sums over them
(_sums_over_trials). So each fit first computes these sums for every session,
once, and from then on works only with them: the model is initialized from
them, and every evaluation of the loss uses them instead of the trials. The
loss and its gradient are the same up to rounding, at a fraction of the cost
(docs/sources.md).

What the fit receives from the data, `trial_sums`, maps each session id to its
sums over the fitted trials, per time bin, in float64:
  A = X'X   (time bins, 9, 9)         X: the 8 variables, plus a column of ones
  c = X'y   (time bins, 9, neurons)   y: the activity of each neuron
  d = y'y   (time bins, neurons)
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---- Model
RANK = 5  # d, number of shared time shapes (Posani-Wang: d = 5 was best)
L2 = 25   # lambda, ridge penalty: of the Posani-Wang grid (25, 75, 200), the value their
          # criterion picks (highest test R²) and that reproduces their trained model

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

    def __init__(self, trial_sums, rank):
        """One U and b per session in `trial_sums` (see the module docstring), and a shared V."""
        super().__init__()
        torch.manual_seed(0)
        n_bins, n_variables = next(iter(trial_sums.values()))[0].shape[:2]
        n_variables -= 1  # the column of ones
        scale = 1 / np.sqrt(n_bins * rank)  # small random start (Posani-Wang code)
        self.V = torch.nn.Parameter(torch.randn(rank, n_bins, dtype=torch.float64) * scale)
        self.U = torch.nn.ParameterDict()
        self.b = torch.nn.ParameterDict()
        for eid, (A, c, _) in trial_sums.items():
            n_neurons = c.shape[2]
            self.U[eid] = torch.nn.Parameter(
                torch.randn(n_neurons, n_variables, rank, dtype=torch.float64) * scale)
            # Start at the average response (~0): the sum of y over trials (c's
            # row for the column of ones) divided by the number of trials (A's
            # last entry). .contiguous() stores the transposed values as a plain
            # copy, which L-BFGS requires.
            self.b[eid] = torch.nn.Parameter((c[:, -1, :] / A[:, -1, -1:]).T.contiguous())

    def beta(self, eid):
        """Time-varying coefficients of a session's neurons: (neurons, variables, time bins)."""
        return self.U[eid] @ self.V

    def predict(self, eid, x):
        """Predicted y (trials, time bins, neurons) from X (trials, time bins, variables)."""
        # Einstein Summation Convention: for each trial k, time bin t and neuron n, sum
        # over the variables v of x[k, t, v] * beta[n, v, t]. Then add the baseline.
        return torch.einsum("ktv,nvt->ktn", x, self.beta(eid)) + self.b[eid].T


# ================================================================ fitting

def _sums_over_trials(session, trials):
    """The sums over one session's trials that its squared error depends on, per time bin, in float64.

    For one neuron and time bin, let x_k be the 8 variables of trial k plus a 1
    (so that the baseline b enters like a coefficient), y_k the neuron's
    activity, and w = (beta, b) its 9 coefficients. Expanding the square and
    summing over trials:

        sum_k (x_k·w - y_k)²  =  w'Aw - 2c'w + d,   with  A = sum_k x_k x_k'  (X'X)
                                                          c = sum_k y_k x_k   (X'y)
                                                          d = sum_k y_k²      (y'y)

    A, c and d depend only on the data, not on w: they are computed here once
    per fit, and every evaluation of the loss (_session_loss) uses them instead
    of going through the trials. A is the same for all of a session's neurons.
    Returns X'X (bins, 9, 9), X'y (bins, 9, neurons) and y'y (bins, neurons).
    """
    x = torch.tensor(session["X"][trials], dtype=torch.float64)    # (trials, bins, 8)
    y = torch.tensor(session["y"][trials], dtype=torch.float64)    # (trials, bins, neurons)
    x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=2)        # (trials, bins, 9)
    return (torch.einsum("ktv,ktw->tvw", x, x),
            torch.einsum("ktv,ktn->tvn", x, y),
            (y ** 2).sum(axis=0))


def _trial_sums(data, trials):
    """`trial_sums` (see the module docstring) for every session in `trials`.

    `data` maps each session id to its inputs (the inputs stage's files),
    `trials` to the trials fitted.
    """
    return {eid: _sums_over_trials(data[eid], trials[eid]) for eid in trials}


def _session_loss(model, eid, session_sums, l2):
    """Squared prediction error plus ridge penalty on the betas, for one session.

    Posani-Wang Methods, eq. 7. The baseline b is not penalized (Posani-Wang code).
    The error is computed from the session's sums over trials, `session_sums`
    = (A, c, d) (_sums_over_trials), as w'Aw - 2c'w + d for each neuron and
    time bin, which equals the sum over trials of its squared prediction errors.
    """
    A, c, d = session_sums
    w = torch.cat([model.beta(eid), model.b[eid][:, None, :]], dim=1)   # (neurons, 9, bins)
    error = (torch.einsum("nvt,tvw,nwt->", w, A, w)
             - 2 * torch.einsum("nvt,tvn->", w, c)
             + d.sum())
    penalty = l2 * (model.beta(eid) ** 2).sum()
    return error + penalty


def fit(model, trial_sums, l2, name):
    """Fit the model to all sessions at once with L-BFGS (Posani-Wang settings).

    `trial_sums` holds each session's sums over the fitted trials (see the
    module docstring). Shows a progress bar labelled `name`. Returns the loss
    after every evaluation, to see how the fit converged.
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
        for eid in trial_sums:
            # Gradient accumulation: one session's loss and gradient at a time.
            loss = _session_loss(model, eid, trial_sums[eid], l2)
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
    trial_sums = _trial_sums(data, {eid: np.arange(len(d["X"])) for eid, d in data.items()})
    model = RRR(trial_sums, RANK)
    losses = fit(model, trial_sums, L2, name="final fit (all trials)")
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
    trial_sums = _trial_sums(data, train)
    model = RRR(trial_sums, RANK)
    losses = fit(model, trial_sums, L2, name=f"split {split + 1}/{N_SPLITS}")
    r2 ={eid: test_r2(model, eid, torch.tensor(d["X"][test[eid]], dtype=torch.float64),
                       d["y"][test[eid]], d["y_mean"], d["y_std"])
          for eid, d in data.items()}
    return r2, losses
