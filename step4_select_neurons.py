"""Step 4: goodness of fit of every neuron, and selection of the selective ones.

For each neuron:
  - the RRR's test R², averaged over step 3's splits;
  - the null model's test R²: it predicts every test trial with the average
    response of the training trials, using the same splits and the same R²;
  - delta R² = R²(RRR) - R²(null): what the 8 variables add beyond the average
    response. Neurons with delta R² above MIN_DELTA_R2 are "selective".

Output: a table with one row per neuron and their coefficients (beta), in
outputs/neurons/; a check plot in outputs/checks/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from step3_fit_rrr import N_SPLITS, r2_spike_counts, split_trials

# ---- Data location
PREPROCESSED_DIR = "outputs/preprocessed"  # step 2's output
RRR_DIR = "outputs/rrr"                    # step 3's output

# ---- Outputs
NEURONS_DIR = "outputs/neurons"  # per-neuron table and coefficients (used by step 5)
CHECKS_DIR = "outputs/checks"    # check plot

# ---- Selection
# Authors' Fig2c.py keeps neurons with delta R² > 0.015 (the paper's Methods say >= 0.015).
MIN_DELTA_R2 = 0.015


# ================================================================ null model

def null_r2(session):
    """Test R² of the null model for each neuron, averaged over step 3's splits.

    The null model predicts every test trial with the average response of the
    training trials (authors' trial_avg_r2). Same splits and same R² as the RRR.
    """
    r2 = []
    for split in range(N_SPLITS):
        train, test = split_trials(session["X"], session["variables"], split)
        # The training trials' average (one value per time bin and neuron), for every test trial.
        y_hat = np.repeat(session["y"][train].mean(axis=0, keepdims=True), len(test), axis=0)
        r2.append(r2_spike_counts(session["y"][test], y_hat, session["y_mean"], session["y_std"]))
    return np.mean(r2, axis=0)


# ================================================================ check

def plot_check(table, path):
    """R² of the RRR against R² of the null model, and the distribution of delta R².

    Like the paper's Fig. 2b, except that Fig. 2b compares with an average per
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


# ================================================================ main

if __name__ == "__main__":
    # 1. Step 3's results: the per-neuron test R² of each split, and the final model.
    splits = [np.load(f"{RRR_DIR}/split{split}.npz") for split in range(N_SPLITS)]
    final = np.load(f"{RRR_DIR}/all_trials.npz")

    # 2. For each session: R² of the RRR and of the null model, and the coefficients.
    tables, betas = [], []
    for path in sorted(Path(PREPROCESSED_DIR).glob("*.npz")):
        eid = path.stem
        session = np.load(path)
        session_table = pd.DataFrame({
            "eid": eid, "pid": session["pid"], "cluster_id": session["cluster_id"],
            "area": session["area"],
            "r2_rrr": np.mean([split[eid] for split in splits], axis=0),
            "r2_null": null_r2(session),
        })
        tables.append(session_table)
        betas.append(final[f"U_{eid}"] @ final["V"])  # (neurons, variables, time bins)
    table = pd.concat(tables, ignore_index=True)
    table["delta_r2"] = table.r2_rrr - table.r2_null
    table["selected"] = table.delta_r2 > MIN_DELTA_R2

    # 3. Save: one row per neuron, and its coefficients in the same order.
    Path(NEURONS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKS_DIR).mkdir(parents=True, exist_ok=True)
    table.to_csv(f"{NEURONS_DIR}/table.csv", index=False)
    np.save(f"{NEURONS_DIR}/beta.npy", np.concatenate(betas).astype(np.float32))

    # 4. Check plot and summary.
    plot_check(table, f"{CHECKS_DIR}/step4_check.png")
    print(f"Neurons: {len(table)}. Mean R²: RRR {table.r2_rrr.mean():.3f}, "
          f"null {table.r2_null.mean():.3f}. Selected (delta R² > {MIN_DELTA_R2}): "
          f"{table.selected.sum()}")
