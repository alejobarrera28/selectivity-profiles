"""The average selectivity of each cortical area to each variable.

For each selected cortical neuron, its selectivity alpha to each variable
(pipeline.profiles.alpha). Areas with enough selected neurons are kept; alpha
is averaged per area and z-scored across areas, separately for each variable.
Same as the Posani-Wang code (Fig2_globally_structured_brain/Fig2c.py).

Run from the repo root: python -m analyses.area_selectivity <dataset>
Output: the figure and its values, in outputs/<dataset>/results/.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.inputs import VARIABLES
from pipeline.profiles import alpha

# ---- Areas
# The 43 cortical areas, in Posani-Wang's hierarchy order (their data/area_list.csv).
CORTICAL_AREAS = "data/area_list.csv"
MIN_NEURONS_PER_AREA = 30  # Posani-Wang code (Fig2c.py): areas with fewer selected neurons are not shown
# Top to bottom of the figure: the Posani-Wang code groups the areas by family, in this
# order, and then reverses it (Fig2c.py). Areas in none of these families go last,
# before reversing.
AREA_FAMILIES = [("SS",), ("VIS",), ("AUD", "GU"), ("RSP",), ("AI",)]

# ---- Figure
N_TOP = 3  # mark the 3 most selective areas of each variable (Posani-Wang's triangles)


# ================================================================ selectivity

def area_order(areas):
    """Posani-Wang's order of the areas, top to bottom of the figure (Fig2c.py)."""
    ordered = []
    for family in AREA_FAMILIES:
        ordered += [a for a in areas if a not in ordered and any(f in a for f in family)]
    ordered += [a for a in areas if a not in ordered]
    return ordered[::-1]


def area_selectivity(table, beta, cortical_areas):
    """Mean |selectivity| of each area to each variable, z-scored across areas.

    `table` and `beta` are the profiles (checkpoint 4); `cortical_areas` in
    hierarchy order. Returns a DataFrame (areas x variables) and the number of
    selected neurons per area.
    """
    keep = (table.selected & table.area.isin(cortical_areas)).to_numpy()
    counts = table[keep].area.value_counts()
    areas = [a for a in cortical_areas if counts.get(a, 0) >= MIN_NEURONS_PER_AREA]  # hierarchy order
    mean = pd.DataFrame(alpha(beta[keep]), columns=VARIABLES).groupby(table[keep].area.to_numpy()).mean()
    mean = mean.loc[area_order(areas)]
    zscored = (mean - mean.mean()) / mean.std(ddof=0)  # per variable (column), across areas
    return zscored, counts.loc[mean.index]


# ================================================================ figure

def plot_area_selectivity(zscored, path):
    """Heatmap of the z-scored mean |selectivity|, with each variable's top areas marked."""
    fig, ax = plt.subplots(figsize=(3.5, 5.6))
    image = ax.imshow(zscored, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_xticks(range(len(VARIABLES)), VARIABLES, rotation=90)
    ax.set_yticks(range(len(zscored)), zscored.index)
    colorbar = fig.colorbar(image, shrink=0.4)
    colorbar.set_label("mean |α| (z-scored)")
    for j, variable in enumerate(VARIABLES):
        for area in zscored[variable].nlargest(N_TOP).index:
            ax.plot(j, zscored.index.get_loc(area), marker="^", color="red", markersize=5)
    ax.set_title("Mean absolute selectivity\nz-scored across areas", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ================================================================ main

if __name__ == "__main__":
    out = f"outputs/{sys.argv[1]}"

    # 1. The profiles' per-neuron table and coefficients, and the cortical areas.
    table = pd.read_csv(f"{out}/profiles/table.csv")
    beta = np.load(f"{out}/profiles/beta.npy")
    cortical_areas = pd.read_csv(CORTICAL_AREAS, header=None)[0].tolist()

    # 2. Mean selectivity per area, z-scored across areas.
    zscored, counts = area_selectivity(table, beta, cortical_areas)

    # 3. Save the values and the figure.
    zscored.assign(selected_neurons=counts).to_csv(f"{out}/results/area_selectivity.csv")
    plot_area_selectivity(zscored, f"{out}/results/area_selectivity.png")
    print(f"Area selectivity: {len(zscored)} areas, {counts.sum()} selected cortical neurons.")
    for variable in VARIABLES:
        top = zscored[variable].nlargest(N_TOP).index
        print(f"  top {N_TOP} for {variable:8s}: {', '.join(top)}")
