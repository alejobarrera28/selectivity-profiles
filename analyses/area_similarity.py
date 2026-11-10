"""How similar the average selectivity of brain areas is, clustered, across the whole brain.

The matrix of similarity of the estimated selectivity between areas is
clustered, to see whether some structure emerges (where the idea comes from:
docs/beyond_cortex.md). The similarity is Posani-Wang's
(Fig2_globally_structured_brain/Fig2d.py): each area's mean alpha per
variable, z-scored across areas separately for each variable, and the cosine
similarity of every pair of areas, on areas with at least 50 selected neurons.
Unlike Posani-Wang, every brain area, not only cortex, and the matrix is
clustered instead of compared with anatomical connectivity (which exists only
for cortex). Each area is labelled with its major division of the brain
(IBL's Cosmos mapping), to see whether the clusters follow anatomy.

Run from the repo root: python -m analyses.area_similarity <dataset>
Output: the figure and its values, in outputs/<dataset>/results/area_similarity/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iblatlas.regions import BrainRegions
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

from pipeline.inputs import VARIABLES
from pipeline.profiles import alpha

# ---- Areas
MIN_NEURONS_PER_AREA = 50  # Posani-Wang code (Fig2d.py): areas with fewer selected neurons are left out

# ---- Clustering (ours: the standard hierarchical clustering of a similarity
# matrix, distance = 1 - cosine similarity; average linkage needs no choice of
# the number of clusters)
LINKAGE = "average"

# ---- Divisions: is the similarity higher within a division than between?
N_PERMUTATIONS = 10000  # shuffles of the areas' division labels, for the p-value
SEED = 0


# ================================================================ similarity

def area_profiles(table, beta):
    """Mean alpha of each area with enough selected neurons, z-scored across areas per variable.

    `table` and `beta` are the profiles stage's output. Returns a DataFrame
    (areas x variables) and the number of selected neurons per area.
    Posani-Wang code (Fig2d.py), over all brain areas.
    """
    keep = table.selected.to_numpy()
    counts = table[keep].area.value_counts()
    areas = counts[counts >= MIN_NEURONS_PER_AREA].index
    mean = pd.DataFrame(alpha(beta[keep]), columns=VARIABLES).groupby(table[keep].area.to_numpy()).mean()
    mean = mean.loc[areas]
    zscored = (mean - mean.mean()) / mean.std(ddof=0)  # per variable (column), across areas
    return zscored, counts.loc[areas]


def similarity(zscored):
    """Cosine similarity of every pair of areas' z-scored profiles (areas x areas)."""
    unit = zscored.to_numpy() / np.linalg.norm(zscored.to_numpy(), axis=1, keepdims=True)
    return pd.DataFrame(unit @ unit.T, index=zscored.index, columns=zscored.index)


def cluster_order(sim):
    """The areas in the order of the hierarchical clustering's leaves, and its linkage."""
    distance = squareform(1 - sim.to_numpy(), checks=False)
    tree = linkage(distance, method=LINKAGE, optimal_ordering=True)
    return sim.index[leaves_list(tree)], tree


def divisions(areas):
    """Major division of the brain of each area (IBL's Cosmos mapping of the Allen atlas)."""
    return pd.Series(BrainRegions().acronym2acronym(np.asarray(areas), mapping="Cosmos"), index=areas)


def within_minus_between(sim, division):
    """Mean similarity of pairs of areas in the same division, minus that of pairs in different ones.

    p-value: the fraction of N_PERMUTATIONS shuffles of the division labels
    with a difference at least as large.
    """
    upper = np.triu_indices(len(sim), k=1)
    values = sim.to_numpy()[upper]
    labels = division.loc[sim.index].to_numpy()

    def difference(labels):
        same = labels[upper[0]] == labels[upper[1]]
        return values[same].mean() - values[~same].mean()

    observed = difference(labels)
    rng = np.random.default_rng(SEED)
    null = np.array([difference(rng.permutation(labels)) for _ in range(N_PERMUTATIONS)])
    return observed, (np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1)


# ================================================================ figure

def plot_area_similarity(sim, tree, division, path):
    """The similarity matrix in the clustering's order, its dendrogram, and each area's division."""
    names = sorted(division.unique())
    colors = dict(zip(names, plt.get_cmap("tab10").colors))
    fig = plt.figure(figsize=(11, 12))
    # Columns: division strip, matrix, room for the area names, colorbar.
    grid = fig.add_gridspec(2, 4, width_ratios=[0.25, 10, 0.7, 0.3], height_ratios=[1.5, 10],
                            wspace=0.02, hspace=0.02)
    # Dendrogram, above the matrix.
    top = fig.add_subplot(grid[0, 1])
    dendrogram(tree, ax=top, color_threshold=0, above_threshold_color="0.4", no_labels=True)
    top.axis("off")
    # Division of each area, left of the matrix.
    strip = fig.add_subplot(grid[1, 0])
    for i, area in enumerate(sim.index):
        strip.add_patch(plt.Rectangle((0, i - 0.5), 1, 1, color=colors[division[area]]))
    strip.set_xlim(0, 1)
    strip.set_ylim(len(sim) - 0.5, -0.5)
    strip.axis("off")
    # The matrix.
    ax = fig.add_subplot(grid[1, 1])
    image = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(sim)), sim.index, rotation=90, fontsize=5)
    ax.set_yticks(range(len(sim)), sim.index, fontsize=5)
    ax.yaxis.tick_right()
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[1, 3]))
    colorbar.set_label("cosine similarity of the areas' z-scored mean |α|")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[n]) for n in names]
    top.legend(handles, names, ncol=len(names), loc="lower center", bbox_to_anchor=(0.5, 1.0),
               fontsize=8, frameon=False, title="major division")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ main

if __name__ == "__main__":
    out = f"outputs/{sys.argv[1]}"
    results = f"{out}/results/area_similarity"
    Path(results).mkdir(parents=True, exist_ok=True)

    # 1. The profiles' per-neuron table and coefficients.
    table = pd.read_csv(f"{out}/profiles/table.csv")
    beta = np.load(f"{out}/profiles/beta.npy")

    # 2. Each area's z-scored mean selectivity, and the similarity of every pair of areas.
    zscored, counts = area_profiles(table, beta)
    sim = similarity(zscored)

    # 3. Clustering, and each area's major division.
    order, tree = cluster_order(sim)
    division = divisions(sim.index)
    observed, p = within_minus_between(sim, division)

    # 4. Save the values and the figure.
    sim.loc[order, order].to_csv(f"{results}/similarity.csv")
    pd.DataFrame({"division": division, "selected_neurons": counts}).loc[order].to_csv(f"{results}/areas.csv")
    plot_area_similarity(sim.loc[order, order], tree, division, f"{results}/area_similarity.png")
    print(f"Area similarity: {len(sim)} areas with at least {MIN_NEURONS_PER_AREA} selected neurons, "
          f"{counts.sum()} neurons.")
    print(f"  areas per division: {division.value_counts().to_dict()}")
    print(f"  mean similarity within a division minus between divisions: {observed:.3f} "
          f"(p = {p:.4f}, {N_PERMUTATIONS} shuffles of the divisions)")
