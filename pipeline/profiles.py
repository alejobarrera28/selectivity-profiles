"""Checkpoint 4: every neuron's goodness of fit and selectivity profile.

For each neuron:
  - the RRR's test R², averaged over the cross-validation splits;
  - the null model's test R²: it predicts every test trial with the average
    response of the training trials, using the same splits and the same R²;
  - delta R² = R²(RRR) - R²(null): what the 8 variables add beyond the average
    response. Neurons with delta R² above MIN_DELTA_R2 are "selective";
  - its selectivity to each variable, alpha (from its coefficients beta).
"""
import numpy as np
import pandas as pd

from pipeline.rrr import N_SPLITS, r2_spike_counts, split_trials

# ---- Selection
# The Posani-Wang code (Fig2c.py) keeps neurons with delta R² > 0.015 (their Methods say >= 0.015).
MIN_DELTA_R2 = 0.015


# ================================================================ goodness of fit

def null_r2(session):
    """Test R² of the null model for each neuron, averaged over the model's splits.

    The null model predicts every test trial with the average response of the
    training trials (Posani-Wang's trial_avg_r2). Same splits and same R² as the RRR.
    """
    r2 = []
    for split in range(N_SPLITS):
        train, test = split_trials(session["X"], session["variables"], split)
        # The training trials' average (one value per time bin and neuron), for every test trial.
        y_hat = np.repeat(session["y"][train].mean(axis=0, keepdims=True), len(test), axis=0)
        r2.append(r2_spike_counts(session["y"][test], y_hat, session["y_mean"], session["y_std"]))
    return np.mean(r2, axis=0)


def neuron_table(data, split_r2):
    """One row per neuron: its session, probe, cluster, area, R², delta R² and selection.

    `data` maps each session id to its inputs (checkpoint 2); `split_r2` is the
    list of {eid: test R² per neuron}, one per split (checkpoint 3).
    """
    table = pd.concat([pd.DataFrame({
        "eid": eid, "pid": session["pid"], "cluster_id": session["cluster_id"],
        "area": session["area"],
        "r2_rrr": np.mean([split[eid] for split in split_r2], axis=0),
        "r2_null": null_r2(session),
    }) for eid, session in data.items()], ignore_index=True)
    table["delta_r2"] = table.r2_rrr - table.r2_null
    table["selected"] = table.delta_r2 > MIN_DELTA_R2
    return table


# ================================================================ selectivity

def alpha(beta):
    """Selectivity of each neuron to each variable: sum over time of |beta(t)|.

    `beta` is (neurons, variables, time bins); returns (neurons, variables).
    Posani-Wang Methods, "Computing the selectivity profiles".
    """
    return np.abs(beta).sum(axis=2)
