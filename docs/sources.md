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
- The IBL data: the Brain Wide Map datasets downloaded by the IBL AI agent
  (https://github.com/int-brain-lab/ibl-ai-agent, commit 03ec7c4), in
  `../ibl-ai-agent/reports/datasets/`: `bwm_ephys` 1.2.1 and `bwm_behavior`
  2.0.0, built from IBL's `2023_12_bwm_release`. 172 of the 178 sessions are
  there. The agent's package is only used to decode their compressed files.
  Its neurons are IBL's 2024 neuron lists (revision 2024-05-06), not the
  original ones Posani-Wang used (see [differences.md](differences.md)).

Why the local data and not IBL's servers: we chose not to download anything
new (Posani-Wang's README estimates ~3 days for the download and training, and the
laptop had ~15 GB of free disk). The price is that the local spike files only
contain good units (see [differences.md](differences.md)). Getting every
cluster, as Posani-Wang did, would mean streaming the spike files of the
sessions' 289 probes from IBL's ONE API (roughly estimated at ~100 GB),
binning each one and deleting it. Only the spikes given to `prepare_neurons`
would change.

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
- Wheel, whisker and lick signals: `get_wheelvelocity`, `get_interpbeh` and
  `get_tongue` in `utils/download_data.py`. The wheel filter and the lick
  detector are IBL's (`brainbox.behavior.wheel.velocity_filtered`,
  `brainbox.behavior.dlc.get_licks`).
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
- Their fit sums all sessions' losses and computes the gradient once, with
  every session in float64; we compute it session by session and add the
  gradients (gradient accumulation), converting each session to float64 only
  while its loss is computed, so the fit fits in 16 GB of memory. Same
  gradients, same fit.
- After choosing the hyperparameters they refit the 3 splits under new file
  names; with fixed hyperparameters we need each split only once.
- The time axis is defined once in the sessions stage and saved with the
  data; later stages read it from there.
