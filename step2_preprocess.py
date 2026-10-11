"""Step 2: turn each session from step 1 into the model's inputs (X) and targets (y).

For each session saved by step 1, two functions build what the RRR model is
fitted on:
  prepare_targets -> y: smoothed spike counts, z-scored per neuron and time bin
  prepare_inputs  -> X: the 8 variables, z-scored per variable and time bin,
                     plus a column of ones for the intercept

Output: one file per session in outputs/preprocessed/, plus a check plot and a
per-session summary in outputs/checks/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ---- Data location
SESSIONS_DIR = "outputs/sessions"  # step 1's output

# ---- Outputs
PREPROCESSED_DIR = "outputs/preprocessed"  # one .npz per session (used by step 3)
CHECKS_DIR = "outputs/checks"              # check plot and per-session summary

# ---- Time axis
DT = 0.01  # seconds per bin, as in step 1

# ---- Z-scoring (y and X)
STD_FLOOR = 1e-8  # authors' code: avoids dividing by 0 when a value never varies across trials

# ---- Targets (y)
SMOOTH_SD = 2  # bins (20 ms): s.d. of the Gaussian smoothing (paper Methods)

# ---- Inputs (X)
# The 8 variables, in the order of Fig. 2 (paper Methods, "Input variables").
VARIABLES = ["block", "side", "contrast", "choice", "outcome", "wheel", "whisker", "lick"]
MOVEMENTS = ["wheel", "whisker", "lick"]  # the ones shifted in time (paper Methods)
# Contrast encoding (paper Methods): 0% -> 0, low (<= 12.5%) -> 1, high (> 12.5%) -> 4.
LOW_CONTRAST_MAX = 0.125
LOW_CONTRAST_CODE, HIGH_CONTRAST_CODE = 1, 4


# ================================================================ helpers

def _zscore_over_trials(x):
    """z-score x (trials, bins, columns) across trials, separately for each bin and column.

    Returns the z-scored array and the mean and std used, each (bins, columns).
    """
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), STD_FLOOR)
    return (x - mean) / std, mean, std


# ================================================================ targets (y)

def prepare_targets(spikes):
    """y: spike counts smoothed in time, then z-scored per neuron and time bin.

    `spikes` is (trials, bins, neurons). Z-scoring puts every neuron and time
    bin on the same scale, so the fit weighs them equally and the coefficients
    can be compared across neurons (paper Methods, eq. 1).
    Returns y, and the mean and std per (bin, neuron) needed to turn the
    model's predictions back into spike counts.
    """
    smoothed = gaussian_filter1d(spikes.astype(float), SMOOTH_SD, axis=1)
    return _zscore_over_trials(smoothed)


# ================================================================ inputs (X)

def _task_variables(session):
    """Block, side, contrast, choice and outcome of each trial.

    Encoded as in the paper's Methods ("Input variables"). For the outcome we
    use IBL's feedbackType; the authors' code computes it as "stimulus side ==
    choice", which gives the same value in every trial.
    """
    contrast = np.nan_to_num(session["contrastLeft"]) + np.nan_to_num(session["contrastRight"])
    return {
        "block": np.where(session["probabilityLeft"] == 0.8, 1, -1),  # +1: left block, p(left) = 0.8
        "side": np.where(np.isnan(session["contrastLeft"]), -1, 1),   # +1: stimulus on the left
        "contrast": np.select([contrast == 0, contrast <= LOW_CONTRAST_MAX],
                              [0, LOW_CONTRAST_CODE], HIGH_CONTRAST_CODE),
        "choice": session["choice"],          # +1: clockwise wheel turn
        "outcome": session["feedbackType"],   # +1: reward
    }


def _best_shift(trace, spikes):
    """Time shift that best lines up a movement trace with the neural activity.

    `trace` is (trials, behaviour bins), `spikes` (trials, neural bins, neurons).
    Cross-correlation, as in the authors' find_bestdelay_byCC: for each shift
    (within the trace's extra bins, +-100 ms), remove each trial's mean, take
    the dot product of the shifted trace with every neuron (summed over time,
    averaged over trials), and combine the neurons as sqrt(sum of squares). If
    the best shift is at the edge of the range, the search counts as failed and
    no shift is used (authors' code). The paper's Methods only describe the
    idea; the details follow the authors' code.

    Returns the shift in bins (positive: the movement comes after the neural
    activity it best matches) and the cross-correlation for every shift.
    """
    n_bins = spikes.shape[1]
    max_shift = (trace.shape[1] - n_bins) // 2  # the trace's extra bins on each side (10)
    trace = trace - trace.mean(axis=1, keepdims=True)
    spikes = spikes - spikes.mean(axis=1, keepdims=True)
    cross_corr = np.array([  # one value per shift, from -max_shift to +max_shift
        np.linalg.norm(np.einsum("kt,ktn->n", trace[:, start:start + n_bins], spikes)) / len(trace)
        for start in range(2 * max_shift + 1)
    ])
    best = int(np.argmax(cross_corr)) - max_shift
    if abs(best) == max_shift:
        best = 0
    return best, cross_corr


def prepare_inputs(session, spikes):
    """X: the 8 variables at every trial and time bin, z-scored, plus a column of ones.

    Task variables are constant within a trial. Movement traces are shifted by
    their best shift (_best_shift) and cut to the neural time bins. Every
    variable is then z-scored per time bin, so that the coefficients are
    comparable across variables (paper Methods). The column of ones lets the
    model fit an intercept, a per-neuron baseline over time (authors' code).
    Returns X (trials, bins, 9) and, per movement, its shift and cross-correlation curve.
    """
    n_trials, n_bins, _ = spikes.shape
    columns = {name: np.repeat(values[:, None], n_bins, axis=1)
               for name, values in _task_variables(session).items()}
    shifts = {}
    for name in MOVEMENTS:
        trace = session[name]
        shift, cross_corr = _best_shift(trace, spikes)
        start = (trace.shape[1] - n_bins) // 2 + shift  # unshifted start (10) + the shift
        columns[name] = trace[:, start:start + n_bins]
        shifts[name] = (shift, cross_corr)
    x = np.stack([columns[name] for name in VARIABLES], axis=2).astype(float)
    x, _, _ = _zscore_over_trials(x)
    return np.concatenate([x, np.ones((n_trials, n_bins, 1))], axis=2), shifts


# ================================================================ check

def plot_check(first_shifts, summary, path):
    """Top: for one session, how well each shift lines up each movement with the neurons.
    Bottom: the shift chosen for each movement across all sessions.
    """
    fig, axes = plt.subplots(2, len(MOVEMENTS), figsize=(12, 6))
    for j, name in enumerate(MOVEMENTS):
        shift, cross_corr = first_shifts[name]
        max_shift = (len(cross_corr) - 1) // 2
        shifts_ms = np.arange(-max_shift, max_shift + 1) * DT * 1000
        axes[0, j].plot(shifts_ms, cross_corr, marker=".")
        axes[0, j].axvline(shift * DT * 1000, color="r", lw=1, label="chosen")
        axes[0, j].set(title=f"{name} (first session)", xlabel="shift (ms)",
                       ylabel="cross-correlation with neurons")
        axes[1, j].hist(summary[f"{name}_shift_ms"], bins=shifts_ms - DT * 500)
        axes[1, j].set(title=f"{name} (all sessions)", xlabel="chosen shift (ms)",
                       ylabel="sessions")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ================================================================ main

if __name__ == "__main__":
    # 1. The sessions saved by step 1.
    paths = sorted(Path(SESSIONS_DIR).glob("*.npz"))
    print(f"Sessions from step 1: {len(paths)}")

    # 2. For each session: build y and X and save them.
    Path(PREPROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKS_DIR).mkdir(parents=True, exist_ok=True)
    summary = []
    for i, path in enumerate(paths):
        # ---- Read step 1's file
        session = np.load(path)
        spikes = session["spikes"]

        # ---- Targets
        y, y_mean, y_std = prepare_targets(spikes)

        # ---- Inputs
        x, shifts = prepare_inputs(session, spikes)

        # ---- Save (not compressed: z-scored decimals barely compress)
        np.savez(
            f"{PREPROCESSED_DIR}/{path.stem}.npz",
            X=x.astype(np.float32),                 # (trials, 100 bins, 8 variables + ones)
            y=y.astype(np.float32),                 # (trials, 100 bins, neurons)
            y_mean=y_mean.astype(np.float32),       # (100 bins, neurons), to undo the z-score
            y_std=y_std.astype(np.float32),
            variables=np.array(VARIABLES),          # names of X's first 8 columns
            area=session["area"], pid=session["pid"], cluster_id=session["cluster_id"],
        )
        shift_ms = {f"{name}_shift_ms": round(shifts[name][0] * DT * 1000) for name in MOVEMENTS}
        # A failed search (best shift at the edge of the range) also gives 0 ms.
        failed = {f"{name}_search_failed":
                  np.argmax(shifts[name][1]) in (0, len(shifts[name][1]) - 1)
                  for name in MOVEMENTS}
        print(f"{i + 1:3d}/{len(paths)} {path.stem}: X {x.shape}, y {y.shape}, shifts (ms) "
              + ", ".join(f"{name} {shift_ms[f'{name}_shift_ms']:+d}" for name in MOVEMENTS))
        summary.append({"eid": path.stem, "trials": y.shape[0], "neurons": y.shape[2],
                        **shift_ms, **failed})

        # ---- Check (the first session's cross-correlation curves go in the check plot)
        if i == 0:
            first_shifts = shifts

    # 3. Summary of the whole step, also saved as a table, and the check plot.
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{CHECKS_DIR}/step2_summary.csv", index=False)
    plot_check(first_shifts, summary, f"{CHECKS_DIR}/step2_check.png")
    print(f"\nPreprocessed {len(summary)} sessions. Failed shift searches (no shift used): "
          + ", ".join(f"{name} {summary[f'{name}_search_failed'].sum()}" for name in MOVEMENTS))
