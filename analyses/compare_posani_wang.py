"""Our profiles against Posani-Wang's trained model, neuron by neuron.

For the `posani_wang` dataset, which reads IBL's data as Posani-Wang had it. Their
trained model gives, for each of their neurons, its IBL id, area, test R², null
R² and coefficients beta; our neurons are matched to theirs by IBL id. One
figure per step of the comparison, each titled with what it shows:
  1_data      same data: neurons per session, and the null model's R² (which
              depends only on the data and the preprocessing)
  2_fit       same fit: test R² and delta R², and which neurons pass the threshold
  3_profiles  same profiles: each variable's selectivity alpha, and each
              neuron's profile shape (cosine similarity, blind to scale)
  4_areas     same area selectivity: the area figure from their model and ours
  5_population same population profiles: for each session and variable, its
              neurons' alpha (cosine similarity, blind to scale)

Run from the repo root: python -m analyses.compare_posani_wang <dataset>
Output: the figures and a table of the numbers, in outputs/<dataset>/results/compare_posani_wang/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyses.area_selectivity import CORTICAL_AREAS, MIN_NEURONS_PER_AREA, N_TOP, area_order, area_selectivity
from pipeline.inputs import VARIABLES
from pipeline.profiles import MIN_DELTA_R2, alpha

# ---- Posani-Wang's trained model: their trained_RRR_model/RRR_selectivity.json
# (repo commit 00d40b9, fetched with Git LFS), next to this repo.
POSANI_WANG_MODEL = "../brainwide-RRR-encoding-model/trained_RRR_model/RRR_selectivity.json"


# ================================================================ profiles

def posani_wang_profiles(path):
    """Their neurons as a table like ours (uuid, eid, area, R², selection), and their beta.

    Their beta has the baseline b appended as a last row (their
    RRRGD_model.compute_beta_m); it is dropped, as in their Fig2c.py.
    """
    model = pd.read_json(path)
    table = pd.DataFrame({"uuid": model.uuids, "eid": model.eid, "area": model.acronym,
                          "r2_rrr": model.RRR_r2, "r2_null": model.null_r2})
    table["delta_r2"] = table.r2_rrr - table.r2_null
    table["selected"] = table.delta_r2 > MIN_DELTA_R2  # the same threshold as ours
    beta = np.array([np.asarray(b)[:-1] for b in model.RRR_beta], dtype=np.float32)
    return table, beta


def match(ours, theirs):
    """Row positions in `ours` and in `theirs` of the neurons both have, by IBL id."""
    position = pd.Series(np.arange(len(theirs)), index=theirs.uuid)
    in_both = ours.uuid.isin(position.index).to_numpy()
    return np.flatnonzero(in_both), position[ours.uuid[in_both]].to_numpy()


def cosine(a, b):
    """Cosine similarity of each row of a with the same row of b: 1 for the same shape, whatever the scale."""
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def population_cosine(alpha_ours, alpha_theirs, eid):
    """Cosine similarity of each session's population profile, per variable (sessions x variables).

    A session's population profile for a variable is the alpha of each of its
    neurons; ours against theirs (suggested by Shuqi Wang).
    """
    sessions = pd.Series(eid).groupby(eid).indices
    return pd.DataFrame({s: cosine(alpha_ours[i].T, alpha_theirs[i].T) for s, i in sessions.items()},
                        index=VARIABLES).T


# ================================================================ figures

def _scatter(ax, x, y, title, threshold=None, differing=None):
    """One dot per neuron (or area), theirs (x) against ours (y), with the diagonal.

    `differing` marks the neurons of the sessions whose neurons differ from
    theirs: drawn in red, and the correlation is also given without them.
    """
    x, y = np.asarray(x), np.asarray(y)
    r = f"r = {np.corrcoef(x, y)[0, 1]:.3f}"
    if differing is None:
        ax.scatter(x, y, s=2, alpha=0.3)
    else:
        ax.scatter(x[~differing], y[~differing], s=2, alpha=0.3)
        ax.scatter(x[differing], y[differing], s=2, alpha=0.5, color="tab:red")
        r += f" ({np.corrcoef(x[~differing], y[~differing])[0, 1]:.3f} without the red)"
    low, high = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([low, high], [low, high], "k--", lw=1)
    if threshold is not None:
        ax.axvline(threshold, color="0.4", lw=1, ls=":")
        ax.axhline(threshold, color="0.4", lw=1, ls=":")
    ax.set(title=f"{title}\n{r}", xlabel="Posani-Wang", ylabel="ours")


def plot_data(counts, ours, theirs, n_theirs, differing, path):
    """Neurons per session, and the null model's R² of every neuron both have (red: the sessions that differ)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    same = (counts.ours == counts.theirs).to_numpy()
    ax = axes[0]
    ax.scatter(counts.theirs[same], counts.ours[same], s=12, label=f"same number ({same.sum()} sessions)")
    ax.scatter(counts.theirs[~same], counts.ours[~same], s=12, color="tab:red",
               label=f"different ({(~same).sum()} sessions)")
    top = counts.values.max()
    ax.plot([0, top], [0, top], "k--", lw=1)
    ax.set(title="Neurons per session", xlabel="Posani-Wang", ylabel="ours")
    ax.legend(fontsize=8)
    _scatter(axes[1], theirs.r2_null, ours.r2_null, "Null model's test R² (depends only on the data)",
             differing=differing)
    fig.suptitle(f"1. Same data: {len(ours)} of their {n_theirs} neurons in these sessions found "
                 f"({len(ours) / n_theirs:.1%}), {(ours.area == theirs.area).mean():.0%} in the same area")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_fit(ours, theirs, differing, path):
    """Test R² and delta R² of every neuron both have, and who passes the selection threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _scatter(axes[0], theirs.r2_rrr, ours.r2_rrr, "Model's test R²", differing=differing)
    _scatter(axes[1], theirs.delta_r2, ours.delta_r2, "delta R² (dotted: selection threshold)",
             MIN_DELTA_R2, differing)
    selection = pd.crosstab(ours.selected, theirs.selected)
    axes[1].text(0.97, 0.03, f"selected by both: {selection.loc[True, True]}\n"
                             f"only by ours: {selection.loc[True, False]}\n"
                             f"only by theirs: {selection.loc[False, True]}\n"
                             f"by neither: {selection.loc[False, False]}",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)
    fig.suptitle(f"2. Same fit: mean test R² {ours.r2_rrr.mean():.3f} (theirs {theirs.r2_rrr.mean():.3f}); "
                 f"{(ours.selected == theirs.selected).mean():.0%} of neurons get the same selection")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_profiles(alpha_ours, alpha_theirs, differing, path):
    """Each variable's alpha, and each neuron's profile shape, on the neurons selected in both."""
    fig, axes = plt.subplots(2, 5, figsize=(21, 7.5))
    slopes = []
    for v, (variable, ax) in enumerate(zip(VARIABLES, axes.flat)):
        slope = np.polyfit(alpha_theirs[:, v], alpha_ours[:, v], 1)[0]
        slope_same = np.polyfit(alpha_theirs[~differing, v], alpha_ours[~differing, v], 1)[0]
        slopes.append(slope)
        _scatter(ax, alpha_theirs[:, v], alpha_ours[:, v],
                 f"alpha, {variable} (slope {slope:.2f}; {slope_same:.2f} without the red)", differing=differing)
    similarity = cosine(alpha_ours, alpha_theirs)
    ax = axes.flat[8]
    ax.hist([similarity[~differing], similarity[differing]], bins=np.linspace(similarity.min(), 1, 61),
            stacked=True, log=True, color=["tab:blue", "tab:red"])  # log: the red are few
    ax.set(title=f"Profile shape per neuron: cosine similarity\nmedian {np.median(similarity):.3f} "
                 f"({np.median(similarity[~differing]):.3f} without the red, "
                 f"{np.median(similarity[differing]):.3f} the red)",
           xlabel="cosine similarity (1: same shape)", ylabel="neurons")
    axes.flat[9].axis("off")
    fig.suptitle(f"3. Same profiles, {np.mean(slopes) - 1:.0%} larger on average: "
                 f"alpha of the {len(alpha_ours)} neurons selected in both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _area_correlation(theirs, ours):
    """Correlation of two area figures, over the areas both show and every variable."""
    common = [a for a in theirs.index if a in ours.index]
    return np.corrcoef(theirs.loc[common].values.ravel(), ours.loc[common].values.ravel())[0, 1]


def plot_population(population, differing, path):
    """Each session's population profile, per variable: one dot per session (red: the sessions that differ)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(population))
    for v, variable in enumerate(VARIABLES):
        ax.scatter(v + jitter[~differing], population[variable][~differing], s=8, alpha=0.5, color="tab:blue")
        ax.scatter(v + jitter[differing], population[variable][differing], s=8, alpha=0.8, color="tab:red")
    ax.set_xticks(range(len(VARIABLES)), VARIABLES)
    ax.set(ylabel="cosine similarity (1: same shape)",
           title=f"median {np.median(population[~differing].values):.3f} in the {(~differing).sum()} sessions "
                 f"with the same neurons, {np.median(population[differing].values):.3f} in the {differing.sum()} red")
    fig.suptitle("5. Same population profiles: per session and variable, its neurons' alpha, ours against theirs")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def shared_top(theirs, ours):
    """How many of each variable's top areas the two area figures share, over all variables."""
    return sum(len(theirs[v].nlargest(N_TOP).index.intersection(ours[v].nlargest(N_TOP).index)) for v in VARIABLES)


def plot_areas(pairs, cortical_areas, path):
    """The area selectivity figure from their model and from ours, side by side, and their correlation.

    One row per (theirs, ours, what both are made from). Each figure has the
    areas either of the two shows, gray where it does not show one (fewer than
    MIN_NEURONS_PER_AREA selected neurons), and its own top areas marked, as in
    analyses.area_selectivity.
    """
    fig, axes = plt.subplots(len(pairs), 3, figsize=(13, 6.8 * len(pairs)), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    colors = plt.get_cmap("Reds").copy()
    colors.set_bad("0.75")
    correlations = []
    for row, (theirs, ours, made_from) in zip(axes, pairs):
        areas = area_order([a for a in cortical_areas if a in theirs.index or a in ours.index])
        for ax, values, title in [(row[0], theirs, "Posani-Wang"), (row[1], ours, "ours")]:
            image = ax.imshow(values.reindex(areas), aspect="auto", cmap=colors, interpolation="nearest",
                              vmin=-2.5, vmax=3.5)
            for j, variable in enumerate(VARIABLES):
                for area in values[variable].nlargest(N_TOP).index:
                    ax.plot(j, areas.index(area), marker="^", color="black", markersize=5)
            ax.set_xticks(range(len(VARIABLES)), VARIABLES, rotation=90)
            ax.set_yticks(range(len(areas)), areas if ax is row[0] else [])  # same rows: names once
            ax.set_title(title, fontsize=10)
        fig.colorbar(image, ax=row[:2], shrink=0.4, label="mean |alpha| (z-scored across areas)")
        common = [a for a in areas if a in theirs.index and a in ours.index]
        x, y = theirs.loc[common].values.ravel(), ours.loc[common].values.ravel()
        correlations.append(f"{np.corrcoef(x, y)[0, 1]:.2f} from {made_from}")
        row[2].set_box_aspect(1)
        _scatter(row[2], x, y, f"From {made_from}: one dot per area and variable\n({len(common)} areas both show; "
                               f"same top {N_TOP}: {shared_top(theirs, ours)} of {N_TOP * len(VARIABLES)})")
    fig.suptitle(f"4. Same area selectivity: r = {', '.join(correlations)}\n"
                 f"(▲ each figure's top {N_TOP}; gray: fewer than {MIN_NEURONS_PER_AREA} selected neurons)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ================================================================ main

if __name__ == "__main__":
    out = f"outputs/{sys.argv[1]}"
    results = f"{out}/results/compare_posani_wang"
    Path(results).mkdir(parents=True, exist_ok=True)

    # 1. Our profiles and Posani-Wang's, and the neurons both have.
    ours = pd.read_csv(f"{out}/profiles/table.csv")
    beta_ours = np.load(f"{out}/profiles/beta.npy")
    theirs, beta_theirs = posani_wang_profiles(POSANI_WANG_MODEL)
    i_ours, i_theirs = match(ours, theirs)
    o, t = ours.iloc[i_ours].reset_index(drop=True), theirs.iloc[i_theirs].reset_index(drop=True)

    # 2. Neurons, session by session, on the sessions both have, and the neurons
    #    of the sessions where they differ.
    sessions = sorted(set(ours.eid) & set(theirs.eid))
    counts = pd.DataFrame({"ours": ours.groupby("eid").size(), "theirs": theirs.groupby("eid").size()}).loc[sessions]
    differing = o.eid.isin(counts.index[counts.ours != counts.theirs]).to_numpy()

    # 3. Selectivity alpha, on the neurons selected in both.
    both = (o.selected & t.selected).to_numpy()
    alpha_ours, alpha_theirs = alpha(beta_ours[i_ours][both]), alpha(beta_theirs[i_theirs][both])
    population = population_cosine(alpha(beta_ours[i_ours]), alpha(beta_theirs[i_theirs]), o.eid.to_numpy())
    population_differing = population.index.isin(counts.index[counts.ours != counts.theirs])

    # 4. The area selectivity figure, from each model: from all their neurons and
    #    all ours, and again from the sessions where both have the same neurons.
    cortical_areas = pd.read_csv(CORTICAL_AREAS, header=None)[0].tolist()
    areas_ours, _ = area_selectivity(ours, beta_ours, cortical_areas)
    areas_theirs, _ = area_selectivity(theirs, beta_theirs, cortical_areas)
    same_sessions = counts.index[counts.ours == counts.theirs]
    in_ours, in_theirs = ours.eid.isin(same_sessions).to_numpy(), theirs.eid.isin(same_sessions).to_numpy()
    areas_ours_same, _ = area_selectivity(ours[in_ours].reset_index(drop=True), beta_ours[in_ours], cortical_areas)
    areas_theirs_same, _ = area_selectivity(theirs[in_theirs].reset_index(drop=True), beta_theirs[in_theirs],
                                            cortical_areas)

    # 5. Save the numbers and the figures.
    numbers = {
        "sessions in both": len(sessions),
        "their neurons in these sessions": int(counts.theirs.sum()),
        "our neurons in these sessions": int(counts.ours.sum()),
        "neurons in both": len(o),
        "sessions with the same neuron count": int((counts.ours == counts.theirs).sum()),
        "same area": float((o.area == t.area).mean()),
        "mean test R², ours": o.r2_rrr.mean(), "mean test R², theirs": t.r2_rrr.mean(),
        "mean null R², ours": o.r2_null.mean(), "mean null R², theirs": t.r2_null.mean(),
        "selected, ours": int(o.selected.sum()), "selected, theirs": int(t.selected.sum()),
        "selected in both": int(both.sum()),
        "selection agreement": float((o.selected == t.selected).mean()),
        "selection agreement, sessions with the same neurons": float((o.selected == t.selected)[~differing].mean()),
        "test R² correlation": np.corrcoef(o.r2_rrr, t.r2_rrr)[0, 1],
        "test R² correlation, sessions with the same neurons":
            np.corrcoef(o.r2_rrr[~differing], t.r2_rrr[~differing])[0, 1],
        "profile cosine similarity, median": float(np.median(cosine(alpha_ours, alpha_theirs))),
        "population profile cosine similarity, median": float(np.median(population.values)),
        "population profile cosine similarity, median, sessions with the same neurons":
            float(np.median(population[~population_differing].values)),
        "areas in the figure, ours": len(areas_ours), "areas in the figure, theirs": len(areas_theirs),
        "area figure correlation (common areas)": _area_correlation(areas_theirs, areas_ours),
        "area figure correlation, sessions with the same neurons": _area_correlation(areas_theirs_same,
                                                                                     areas_ours_same),
        "area figure, same top areas": shared_top(areas_theirs, areas_ours),
        "area figure, same top areas, sessions with the same neurons": shared_top(areas_theirs_same, areas_ours_same),
        **{f"alpha {v} correlation": np.corrcoef(alpha_ours[:, i], alpha_theirs[:, i])[0, 1]
           for i, v in enumerate(VARIABLES)},
        **{f"alpha {v} slope (ours on theirs)": np.polyfit(alpha_theirs[:, i], alpha_ours[:, i], 1)[0]
           for i, v in enumerate(VARIABLES)},
    }
    pd.Series(numbers, name="value").to_csv(f"{results}/numbers.csv")
    plot_data(counts, o, t, int(counts.theirs.sum()), differing, f"{results}/1_data.png")
    plot_fit(o, t, differing, f"{results}/2_fit.png")
    plot_profiles(alpha_ours, alpha_theirs, differing[both], f"{results}/3_profiles.png")
    plot_areas([(areas_theirs, areas_ours, "all the neurons"),
                (areas_theirs_same, areas_ours_same, f"the {len(same_sessions)} sessions with the same neurons")],
               cortical_areas, f"{results}/4_areas.png")
    plot_population(population, population_differing, f"{results}/5_population.png")
    for name, value in numbers.items():
        print(f"  {name:40s} {value:.3f}" if isinstance(value, float) else f"  {name:40s} {value}")
