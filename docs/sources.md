# Sources

Where every input comes from, and where each rule lives in the Posani-Wang code.
Comments in the code say whether each rule comes from the Posani-Wang Methods or
the Posani-Wang code; this file says where to find it.

- Paper ("Posani-Wang"): Posani, Wang, Muscinelli, Paninski & Fusi, "Rarely categorical,
  highly separable representations along the cortical hierarchy", Nature
  (2026), https://doi.org/10.1038/s41586-026-10668-4.
- Code: https://github.com/realwsq/brainwide-RRR-encoding-model,
  commit 00d40b9 (2026-09-24).

## Inputs

- `data/eid_list.npy`: the 178 sessions Posani-Wang used. Copied unchanged from
  their repo (`data/eid_list.npy`). We use their list instead of rebuilding it
  because the same search on today's IBL database would return different
  sessions.
- `data/area_list.csv`: the 43 cortical areas Posani-Wang analyses, in hierarchy
  order. Copied unchanged from the same place. Both files: Copyright (c) 2026
  Shuqi Wang, MIT License.
- The IBL data: read from IBL's public server
  (https://openalyx.internationalbrainlab.org) with IBL's ONE, each file in its
  last revision dated on or before the dataset's date (`as_of` in `run.py`), the
  same files the Posani-Wang code
  reads. Each session's files are downloaded, read and deleted
  (`pipeline/ibl.py`); only the spike files used are downloaded (times and
  which neuron fired them), about 230 MB per probe.

The date of Posani-Wang's data. IBL revises its files and keeps every version,
each in a folder named by the revision's date. Posani-Wang's trained model gives
each neuron's IBL id: across their sessions, their neurons come from the
original neuron lists, except one probe from a revision of 2022-08-04, and none
from the revisions of 2024-03-22 (50 sessions). So they downloaded the data
between those two dates. In between, IBL revised a few of the files we use: the
neuron tables (`clusters.metrics`: firing rate and quality label) of 32
sessions, and the licks, camera times, motion energy or trials of 14. With the
last day before 2024-03-22, the pipeline reproduces their neurons exactly in the
sessions tested (429 in 07dc4b76, 282 in 875c1e5c, 452 in cde63527, whose
neuron table was revised on 2024-03-12). Their behaviour can't be checked: the
trained model doesn't store it.

One exception: in 18 of their sessions, all from the Häusser lab, IBL later
flagged the wheel position they used as faulty (QC "CRITICAL") and replaced it,
in a 2025 "wheel patch" whose revision is named 2024-05-06 (revisions are named
by date, but not always the date they were made). ONE doesn't serve files
flagged CRITICAL, so for the `shuqi` dataset these sessions have no wheel
position; they are excluded by hand, each with its reason, in
`data/shuqi_excluded_sessions.csv`. With IBL's current data (`mine`) they have
the corrected wheel and are kept.

## The Posani-Wang code

- Trial rules (choice, reaction and response time): `filter_trials_func` in
  `utils/download_data.py`, called from `step1_IBL_downloaddata.py`.
- Session firing-rate filter (0.5-50 Hz): `filter_neurons_func` in
  `utils/download_data.py`, called from `step1_IBL_downloaddata.py`.
- 50/50 block removal, the 100-trial minimum, the neuron rules (area, task
  rate, silent trials), the 5-neuron minimum, and whisker = max of the two
  cameras: `_read_Xy` in `utils/save_and_load_data.py`.
- Time window (they download -0.3 to 0.9 s and crop the spikes to -0.2 to
  0.8 s): `step1_IBL_downloaddata.py` and `_read_Xy`.
- Reading the data: `load_data_from_pid` and `load_spikes_from_eid` in
  `utils/download_data.py` (IBL's `SessionLoader` for trials and motion energy,
  `SpikeSortingLoader` for spikes, the wheel and lick objects), with the Beryl
  mapping in `_append_clusters_data`.
- Wheel, whisker and lick signals: `get_wheelvelocity`, `get_interpbeh` and
  `get_tongue` in `utils/download_data.py`. The wheel's interpolation and
  filter are IBL's (`brainbox.behavior.wheel.interpolate_position`,
  `velocity_filtered`).
- Inputs (smoothing, z-scoring, task variable encoding): `_read_Xy` in
  `utils/save_and_load_data.py`. Movement shifts: `find_bestdelay_byCC`
  (lines 227-247, including the failure rule at 242-246) and
  `_preprocess_movement` (line 89: failed search -> unshifted) in the same
  file. the Posani-Wang code computes the outcome as "stimulus side matches the choice";
  we take IBL's feedbackType, identical in every trial.
- Model: the model, its initial values, the loss and the ridge penalty are
  `RRRGD_model` in `utils/RRRGD.py`. The L-BFGS fit and the cross-validation
  (70/30 splits, seeds 42 + split, R² on held-out trials) are
  `train_model_hyper_selection` in `utils/RRRGD_main_CV.py`, called with the
  grid and the stratification order (block x contrast, block, none) from
  `step2_train_RRR.py`. R² per neuron, un-z-scored and clipped at 0:
  `compute_R2_RRRGD` in `utils/RRRGD.py` and `compute_R2_main` in
  `utils/utils.py`.
- Profiles: the null model's R² is `trial_avg_r2` in
  `utils/analyze_perf_utils.py`, called from `step3_save_trainedRRR_2_df.py`,
  which also builds their per-neuron table (`RRR_r2`, `null_r2`, `RRR_beta`).
  The delta R² > 0.015 selection is in `Fig2_globally_structured_brain/Fig2c.py`.
- Area selectivity (Fig. 2c): `Fig2_globally_structured_brain/Fig2c.py` (cortical neurons with
  delta R² > 0.015, areas with at least 30, alpha = sum over time of |beta|,
  area mean, z-score per variable across areas, the area order, top 3 marked).

Implementation differences that give the same results:

- `_read_Xy` appends a column of ones to X, so that their model's baseline
  (`RRR_b`) is the coefficient of that column. Our model adds the baseline
  explicitly.
- Their fit computes the squared error trial by trial in every evaluation;
  we compute it from sums over trials, once per fit (for each session and
  time bin, X'X, X'y and y'y, with a column of ones in X for the baseline):
  with w = (beta, b), the sum over trials of (x·w - y)² equals
  w'(X'X)w - 2 w'(X'y) + y'y. The loss and its gradient are the same up to
  rounding (checked on 20 sessions fitted to convergence: same number of
  evaluations, test R² within 1e-12, beta within 1e-11), and each evaluation
  no longer goes through every trial, so the fit is about ten times faster.
  Each session's loss and gradient are computed one at a time and added
  (gradient accumulation), as their single sum would give.
- After choosing the hyperparameters they refit the 3 splits under new file
  names; with fixed hyperparameters we need each split only once.
- The time axis is defined once in the sessions stage and saved with the
  data; later stages read it from there.
