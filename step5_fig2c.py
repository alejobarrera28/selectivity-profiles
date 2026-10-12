"""Step 5: Fig. 2c, the average selectivity of each cortical area to each variable.

For each selected cortical neuron (step 4), its selectivity to each variable is
alpha = sum over time of |beta(t)| (paper Methods, "Computing the selectivity
profiles"). Areas with enough selected neurons are kept; alpha is averaged per
area and z-scored across areas, separately for each variable. Same as the
authors' Fig2_globally_structured_brain/Fig2c.py.

Output: the figure and its values in outputs/figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- Data location
NEURONS_DIR = "outputs/neurons"  # step 4's output

# ---- Outputs
FIGURES_DIR = "outputs/figures"

# ---- Areas
# The 43 cortical areas, in the paper's hierarchy order (authors' data/area_list.csv).
CORTICAL_AREAS = pd.read_csv("data/area_list.csv", header=None)[0].tolist()
MIN_NEURONS = 30  # authors' Fig2c.py: areas with fewer selected neurons are not shown
# Top to bottom of the figure: the authors group the areas by family, in this order,
# and then reverse it (Fig2c.py). Areas in none of these families go last, before reversing.
AREA_FAMILIES = [("SS",), ("VIS",), ("AUD", "GU"), ("RSP",), ("AI",)]

# ---- Variables
# Same order as beta's variable axis (step 2's VARIABLES).
VARIABLES = ["block", "side", "contrast", "choice", "outcome", "wheel", "whisker", "lick"]
N_TOP = 3  # mark the 3 most selective areas of each variable (paper's triangles)


# ================================================================ selectivity

def area_order(areas):
    """The authors' order of the areas, top to bottom of the figure (Fig2c.py)."""
    ordered = []
    for family in AREA_FAMILIES:
        ordered += [a for a in areas if a not in ordered and any(f in a for f in family)]
    ordered += [a for a in areas if a not in ordered]
    return ordered[::-1]


def area_selectivity(table, beta):
    """Mean |selectivity| of each area to each variable, z-scored across areas.

    `table` and `beta` are step 4's outputs. Returns a DataFrame (areas x
    variables) and the number of selected neurons per area.
    """
    alpha = np.abs(beta).sum(axis=2)  # (neurons, variables): sum over time of |beta(t)|
    keep = (table.selected & table.area.isin(CORTICAL_AREAS)).to_numpy()
    counts = table[keep].area.value_counts()
    areas = [a for a in CORTICAL_AREAS if counts.get(a, 0) >= MIN_NEURONS]  # hierarchy order
    mean = pd.DataFrame(alpha[keep], columns=VARIABLES).groupby(table[keep].area.to_numpy()).mean()
    mean = mean.loc[area_order(areas)]
    zscored = (mean - mean.mean()) / mean.std(ddof=0)  # per variable (column), across areas
    return zscored, counts.loc[mean.index]


# ================================================================ figure

def plot_fig2c(zscored, path):
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
    # 1. Step 4's per-neuron table and coefficients.
    table = pd.read_csv(f"{NEURONS_DIR}/table.csv")
    beta = np.load(f"{NEURONS_DIR}/beta.npy")

    # 2. Mean selectivity per area, z-scored across areas.
    zscored, counts = area_selectivity(table, beta)

    # 3. Save the values and the figure.
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    zscored.assign(selected_neurons=counts).to_csv(f"{FIGURES_DIR}/fig2c_values.csv")
    plot_fig2c(zscored, f"{FIGURES_DIR}/fig2c.png")
    print(f"Fig. 2c: {len(zscored)} areas, {counts.sum()} selected cortical neurons.")
    for variable in VARIABLES:
        top = zscored[variable].nlargest(N_TOP).index
        print(f"  top {N_TOP} for {variable:8s}: {', '.join(top)}")
