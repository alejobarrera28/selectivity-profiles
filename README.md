# Replicating Fig. 2c from scratch

Goal: replicate Fig. 2c of Posani, Wang et al., "Rarely categorical, highly
separable representations along the cortical hierarchy", Nature 2026
(https://doi.org/10.1038/s41586-026-10668-4). We follow the authors' code
(github.com/realwsq/brainwide-RRR-encoding-model), but rebuild it step by step.

Fig. 2c shows, for each cortical area, how strongly its neurons respond on
average to each of 8 variables (block, side, contrast, choice, outcome, wheel,
whisker, lick). Those strengths come from a reduced-rank regression (RRR)
encoding model fitted to every neuron.

## Status

1. Build sessions: select trials and neurons, bin spikes and behaviour. Done.
2. Preprocessing: smooth and z-score the spikes, encode the task variables,
   shift and z-score the movement traces. To do.
3. Fit the RRR model (one model for all sessions, shared temporal bases). To do.
4. Export the fitted coefficients per neuron, with the goodness of fit. To do.
5. Fig. 2c: per-area mean absolute selectivity, z-scored across areas. To do.

## Running it

Needs the IBL AI agent's repo next to this one (`../ibl-ai-agent`), with its
datasets downloaded. Python and packages are managed with `uv`:

    uv sync                                   # create the environment (once)
    uv run python step1_build_sessions.py     # step 1, about 2.5 minutes

`pyproject.toml` was created with these commands (`uv.lock` and `.venv/` are
generated from it automatically):

    uv init --bare --python 3.12
    uv add "brainwidemap @ git+https://github.com/int-brain-lab/paper-brain-wide-map.git@de37454"
    uv add --editable ../ibl-ai-agent
    uv add numpy pandas pyarrow scipy matplotlib scikit-learn torch

`ibl-ai-agent` is only used to read the agent's compressed data files.
`brainwidemap` is needed because the agent depends on it, and it is only on GitHub.

## Files

- `step1_build_sessions.py`: step 1.
- `data/`: the two lists copied from the authors (see Inputs).
- `outputs/sessions/`: one file per session, made by step 1, used by step 2.
- `outputs/checks/`: plots and tables for checking each step.

## Inputs

- `data/eid_list.npy`: the 178 sessions the authors used. Copied unchanged from
  their repo (data/eid_list.npy, commit 00d40b9, Sept 2026). We use their list
  instead of rebuilding it because the same search on today's IBL database
  would return different sessions.
- `data/area_list.csv`: the 43 cortical areas the paper analyses. Copied
  unchanged from the same place. Both files: Copyright (c) 2026 Shuqi Wang,
  MIT License.
- The IBL data: the copies the IBL AI agent already downloaded, in
  `../ibl-ai-agent/reports/datasets/` (bwm_ephys 1.2.1, bwm_behavior 2.0.0,
  built from IBL's 2023_12_bwm_release). 172 of the 178 sessions are there.

Why the local data and not IBL's servers: we chose not to download anything
new (the authors estimate ~3 days for their download and training, and the
laptop had ~15 GB of free disk). The price is that the local spike files only
contain good units (see Differences). Getting every cluster, as the authors
did, would mean streaming the spike files of the sessions' 289 probes from
IBL's ONE API (roughly estimated at ~100 GB), binning each one and deleting it.
Only the spikes given to `prepare_neurons` would change.

## Step 1: build sessions

`step1_build_sessions.py` goes through the authors' sessions and, for each one:

- Trials (`prepare_trials`): keep trials where the mouse made a choice, moved
  after the stimulus, and started and finished its response within 0.8 s;
  drop the 50/50 block. Sessions with fewer than 100 trials left are skipped.
- Neurons (`prepare_neurons`): count spikes in 100 bins of 10 ms, from -0.2 to
  0.8 s around the stimulus. Keep neurons in a real brain area, with a session
  firing rate between 0.5 and 50 Hz, a rate above 0.5 Hz in that window, and
  spikes in at least half of the trials. Units with no area label in the local
  data are what the authors' atlas mapping calls "root", so they are dropped.
  All brain areas are kept, not only cortex: the RRR model's temporal bases
  are shared by every neuron, so the authors fit it on all neurons and only
  pick cortex for the figure. Sessions with fewer than 5 neurons are skipped.
- Behaviour (`prepare_behaviour`): wheel velocity, whisker motion energy (max
  of the two cameras) and lick counts, in 120 bins of 10 ms from -0.3 to
  0.9 s. The extra 10 bins on each side let step 2 shift each movement trace
  in time to line it up with the neurons.

Each session is saved to `outputs/sessions/<eid>.npz`, with raw values only:

- `spikes`: spike counts, (trials, 100 bins, neurons).
- `wheel`, `whisker`, `lick`: (trials, 120 bins).
- `trial_id`, `probabilityLeft`, `contrastLeft`, `contrastRight`, `choice`,
  `feedbackType`: one value per trial, as in IBL's trials table. A missing
  (NaN) contrast means no stimulus on that side; choice +1 is a clockwise
  wheel turn and -1 counter-clockwise; feedbackType +1 is a reward.
- `area`, `pid`, `cluster_id`: one value per neuron (Beryl area acronym,
  probe id, IBL cluster id).

Results: 170 of the 172 sessions pass (2 have fewer than 100 trials), with
60,528 trials and 19,728 neurons, of which 4,233 are in cortex. The per-session
counts are in `outputs/checks/step1_summary.csv`. The plot
`outputs/checks/step1_check.png` shows the first session's signals averaged by
choice: the wheel turns in opposite directions for the two choices shortly
after the stimulus, and licks follow the response, as expected. numpy prints
one harmless warning: in a few sessions one camera never tracks a tongue tip
reliably, so that camera finds no licks for it; the other camera still does.

How it was checked: on the first session (875c1e5c…), we rewrote the authors'
trial and neuron selection line by line from their code (their binning,
cropping and tests) in a throwaway script. It gave the same 535 trials and the
same 48 neurons as our functions.

## Differences from the authors

- Neurons: we only have "good units" (IBL's best-quality neurons), while the
  authors used every neuron firing between 0.5 and 50 Hz. In the authors'
  sessions that's about 5,800 cortical candidates instead of 39,000; after
  step 1 we have 4,233 cortical neurons, while the paper reports about 14,000.
  So fewer areas will reach the 30 well-fit neurons needed to appear in the
  figure. This is by far the largest difference.
- Sessions: 172 instead of 178, because 6 are missing from the local data.
- Minimum of 100 trials per session: we count the trials left after all
  selection rules. The authors' code counts before removing the 50/50 block,
  which looks like an accident (the model is only fitted on the trials left
  after it). This drops 2 sessions they would keep (about 20 cortical neurons).
- Licks: the authors used IBL's lick times. The local data doesn't include
  them, so we recompute them with IBL's own method from the tongue keypoints.
  The keypoints come from a newer tracker (Lightning Pose instead of
  DeepLabCut), and the right camera is stored at 60 instead of 150 frames per
  second, so our lick times will differ somewhat.
- Wheel velocity: the authors filter the wheel position trial by trial; we
  filter the whole session at once with the same filter, which avoids edge
  effects at the start and end of each trial. Small difference.

## What to compare our results against

- The paper: about 14,000 cortical neurons after selection, of which 4,617
  are fitted well enough by the model (ΔR² above 0.015). Fig. 2c shows 31 areas.
- The authors' repo has its own Fig. 2c output
  (`Fig2_globally_structured_brain/abs_sel_per_area.pdf`), made from their
  pre-trained model. It doesn't exactly match the published figure: it has
  VISC instead of FRP, and a few of the top-3 markers differ (e.g. for "side").
  Their README says IBL data changes across versions and offers the exact data
  on request (shuqi.wang@epfl.ch). So an exact match isn't expected; the
  overall pattern is what to compare.
- Their pre-trained model (`trained_RRR_model/RRR_selectivity.json`, 1.27 GB,
  stored with Git LFS) can be downloaded to compare coefficients neuron by
  neuron, for the neurons we share.

## Where "authors' code" points to

Comments in our code say whether each rule comes from the paper's Methods or
the authors' code. In their repo (commit 00d40b9), the code is here:

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
