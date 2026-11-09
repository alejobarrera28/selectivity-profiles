# Beyond cortex: this research's decisions

The validation is done: on the `shuqi` dataset, our implementation reproduces
Posani-Wang's trained model ([pipeline.md](pipeline.md), Validation against
Posani-Wang's model). From here on, the work is our own: the same engine
applied to the whole brain, starting with the `mine` dataset. This file
records the decisions taken for it, and with whom and when, and the choices
we keep from Posani-Wang for now even though they were made for cortex,
each with how to check it on the new data. [differences.md](differences.md)
stays as the record of the validation (where we differ from Posani-Wang and
why); its limitations of the method still apply here.

## Decided

With Liam Paninski (Slack, 2026-11-08):

- Sessions: every session of the IBL Brain Wide Map, unless Shuqi Wang
  objects. Posani-Wang selected sessions by cortical targets (at most 30 per
  cortical area), so other regions appear only where a probe happened to
  cross them. The list is still to be built from IBL's official Brain Wide
  Map list (about 459 sessions); `mine` still reads Posani-Wang's 178
  (`data/eid_list.npy`).
- Neurons: IBL's good units only (the ones that pass all its quality
  metrics), instead of every unit as in Posani-Wang.
- Data version: IBL's data as of a fixed date (`as_of` in `run.py`,
  2026-11-01), which can be moved later.
- First analyses: decoding and basic clustering.
- The 11 sessions that differ from Posani-Wang (differences.md, Open
  questions): report what we found to Shuqi Wang, and ask Olivier at IBL if
  it stays unresolved. They don't block the work.

## Kept from Posani-Wang for now

Liam wasn't asked about these. We keep Posani-Wang's choice, and revisit it if
the check shows a problem on the `mine` data.

- Firing-rate filter, 0.5-50 Hz (`MIN_FR`, `MAX_FR` in
  `pipeline/sessions.py`). In cortex the upper limit mostly removes
  multi-unit noise; elsewhere it removes whole cell classes with high rates
  (cerebellar Purkinje cells, SNr and GPe neurons, some thalamic and
  brainstem cells), and with good units only, those are already single
  neurons. Check: how many good units each region loses to each neuron rule.
- Good units vary by region: the fraction of units IBL labels good differs
  between regions, so the good-units decision samples regions unevenly.
  Check: the fraction of good units per region.
- One temporal basis shared by all neurons (the d = 5 time shapes of the
  model, `RANK` in `pipeline/rrr.py`). It is learned from all neurons
  together, so the regions with the most neurons dominate it; regions with
  other dynamics (tonic, slow ramps, very fast transients) may be fitted
  worse. Check: test R² and the fraction of selected neurons per region;
  if some regions stand out low, fit a separate basis per group of regions
  and compare.
- One movement shift per session (differences.md, Limitations): a probe often
  crosses cortex, hippocampus and thalamus, which have very different
  latencies, so the shared shift fits some of them badly. Check: the best
  shift per neuron, and how it spreads by region.
- The 8 variables and the -0.2 to 0.8 s window (`VARIABLES` in
  `pipeline/inputs.py`) were chosen for cortical task coding. Other regions
  may encode what the model doesn't have (reward expectation, arousal or
  pupil, paw or body movement) or respond later (late feedback responses in
  dopaminergic areas). Posani-Wang notes that categorical structure depends
  on the chosen variables. Check: the regions where the model explains least.
- Selection, delta R² > 0.015 (`MIN_DELTA_R2` in `pipeline/profiles.py`). It
  is relative to each neuron's variance, and regions differ in rates and
  variability, so the fraction selected differs by region. Posani-Wang warns
  that poorly fitted neurons inflate categorical structure, and that excluding
  too many depletes the centre of the distribution. Check: the fraction
  selected per region.
- The noise floor of alpha (differences.md, Limitations) is higher in regions
  with fewer trials or noisier neurons, and can look like structure. Check:
  alpha with each variable shuffled across trials, per region.

## For the clustering

From the Posani-Wang paper and code; to settle when the clustering analysis
is written.

- Their clustering needs at least 50 selected neurons per region and removes
  clusters dominated by one session. Regions outside cortex are often
  recorded in few sessions, so clusters driven by the session are a real
  risk.
- A region label that lumps distinct nuclei (common in thalamus and midbrain)
  can give clusters that are sub-regions, not categories of neurons.
- Cell types are more discrete in some regions (striatal projection neurons
  and interneurons, Purkinje and nuclear cells): categorical structure there
  can be real but come from the cell type rather than the task, which
  Posani-Wang's Discussion mentions.
- Area-level comparisons: the area selectivity figure z-scores each variable
  across the areas shown, so adding regions changes the colours of the
  cortical ones. Posani-Wang's organizing axis, the cortical hierarchy from
  the Allen connectivity data, doesn't exist outside cortex; another axis
  (brain-wide modules, connectivity to cortex) or none is needed.

## Not a priority

Ideas from the validation, not pursued now: Posani-Wang's Fig. 2d (selectivity
similarity against anatomical connectivity, which needs a permutation test,
since area pairs aren't independent); a bootstrap over neurons of the area
selectivity figure, to see which top areas are robust; alpha-diversity
(Posani-Wang's Fig. 4), which only makes sense with the clustering.
