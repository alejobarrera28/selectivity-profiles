"""Figures to check each checkpoint of the main pipeline, for one dataset.

  signals_by_choice  sessions: population rate, wheel, whisker and licks of one session, by choice
  movement_shifts    inputs: how each movement trace was shifted in time to match the neurons
  fit                model: the final fit's loss and time shapes, and the test R²
  r2_vs_null         profiles: R² of the RRR against the null model, and delta R²

run.py draws each one as soon as its stage is done (check_sessions, ...). To
redraw them all, run from the repo root: python -m analyses.checks <dataset>
Output: the figures, in outputs/<dataset>/results/pipeline/, next to run.py's
summary tables.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.inputs import MOVEMENTS
from pipeline.profiles import MIN_DELTA_R2
from pipeline.rrr import N_SPLITS


# ================================================================ sessions

def plot_signals_by_choice(session, path):
    """Trial-averaged population rate, wheel, whisker and licks, split by choice.

    `session` is one session's file from the sessions stage. If binning and alignment
    are right, the wheel turns in opposite directions for the two choices
    shortly after the stimulus, and licks follow the response.
    """
    dt, spikes = float(session["dt"]), session["spikes"]
    t_neural = float(session["neural_start"]) + dt * (np.arange(spikes.shape[1]) + 0.5)  # spike counts: bin centres
    t_behav = float(session["behav_start"]) + dt * np.arange(session["wheel"].shape[1])  # behaviour: sampled at bin starts
    cw = session["choice"] == 1
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
    signals = [("population rate (Hz/neuron)", t_neural, spikes.mean(axis=2) / dt)]
    signals += [(name, t_behav, session[name]) for name in ("wheel", "whisker", "lick")]
    for ax, (name, t, x) in zip(axes, signals):
        ax.plot(t, x[cw].mean(0), label="choice +1 (clockwise)")
        ax.plot(t, x[~cw].mean(0), label="choice -1 (counter-clockwise)")
        ax.axvline(0, color="k", lw=0.5)
        ax.set(title=name, xlabel="time from stimulus onset (s)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_sessions(out, results):
    """The signals by choice of the first session kept (`out`: the dataset's outputs folder)."""
    kept = pd.read_csv(f"{results}/sessions.csv").query("status == 'saved'")
    plot_signals_by_choice(np.load(f"{out}/sessions/{kept.eid.iloc[0]}.npz"),
                           f"{results}/signals_by_choice.png")


# ================================================================ inputs

def plot_movement_shifts(first, summary, path):
    """Top: for one session, how well each shift lines up each movement with the neurons.
    Bottom: the shift chosen for each movement across all sessions.

    `first` is one session's file from the inputs stage, `summary` the inputs
    stage's table of chosen shifts.
    """
    dt = float(first["dt"])
    fig, axes = plt.subplots(2, len(MOVEMENTS), figsize=(12, 6))
    for j, name in enumerate(MOVEMENTS):
        cross_corr = first[f"{name}_cross_corr"]
        max_shift = (len(cross_corr) - 1) // 2
        shifts_ms = np.arange(-max_shift, max_shift + 1) * dt * 1000
        axes[0, j].plot(shifts_ms, cross_corr, marker=".")
        axes[0, j].axvline(summary[f"{name}_shift_ms"].iloc[0], color="r", lw=1, label="chosen")
        axes[0, j].set(title=f"{name} (first session)", xlabel="shift (ms)",
                       ylabel="cross-correlation with neurons")
        axes[1, j].hist(summary[f"{name}_shift_ms"], bins=shifts_ms - dt * 500)
        axes[1, j].set(title=f"{name} (all sessions)", xlabel="chosen shift (ms)",
                       ylabel="sessions")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_inputs(out, results):
    """The first session's shift curves, and every session's chosen shifts."""
    first = np.load(sorted(Path(f"{out}/inputs").glob("*.npz"))[0])
    plot_movement_shifts(first, pd.read_csv(f"{results}/shifts.csv"),
                         f"{results}/movement_shifts.png")


# ================================================================ model

def plot_fit(final, splits, dt, neural_start, path):
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


def check_model(out, results):
    """The split fits and the final fit (time axis from the inputs)."""
    first = np.load(sorted(Path(f"{out}/inputs").glob("*.npz"))[0])
    splits = [np.load(f"{out}/model/split{split}.npz") for split in range(N_SPLITS)]
    plot_fit(np.load(f"{out}/model/all_trials.npz"), splits,
             float(first["dt"]), float(first["neural_start"]), f"{results}/fit.png")


# ================================================================ profiles

def plot_r2_vs_null(table, path):
    """R² of the RRR against R² of the null model, and the distribution of delta R².

    Like Posani-Wang's Fig. 2b, except that their Fig. 2b compares with an average per
    task condition, and this with the null model (one average over all trials).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(table.r2_null, table.r2_rrr, s=1, alpha=0.3)
    top = max(table.r2_null.max(), table.r2_rrr.max())
    axes[0].plot([0, top], [0, top], "k--", lw=1)
    axes[0].set(xlabel="R² of the null model (average over all trials)",
                ylabel="R² of the RRR",
                title=f"mean R²: RRR {table.r2_rrr.mean():.3f}, null {table.r2_null.mean():.3f}")
    axes[1].hist(table.delta_r2, bins=100)
    axes[1].axvline(MIN_DELTA_R2, color="r", lw=1, label=f"threshold {MIN_DELTA_R2}")
    axes[1].set(xlabel="delta R²", ylabel="neurons", yscale="log",
                title=f"{table.selected.sum()} of {len(table)} neurons selected")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_profiles(out, results):
    """The per-neuron table's R², against the null model's."""
    plot_r2_vs_null(pd.read_csv(f"{out}/profiles/table.csv"), f"{results}/r2_vs_null.png")


# ================================================================ main

if __name__ == "__main__":
    out = f"outputs/{sys.argv[1]}"
    results = f"{out}/results/pipeline"

    # Every check, from the checkpoints already saved.
    for check in (check_sessions, check_inputs, check_model, check_profiles):
        check(out, results)
    print(f"Check figures saved in {results}/")
