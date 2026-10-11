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
   shift and z-score the movement traces. Done.
3. Fit the RRR model (one model for all sessions, shared temporal bases). Done.
4. Export the fitted coefficients per neuron, with the goodness of fit. To do.
5. Fig. 2c: per-area mean absolute selectivity, z-scored across areas. To do.

## Running it

Needs the IBL AI agent's repo next to this one (`../ibl-ai-agent`), with its
datasets downloaded. Python and packages are managed with `uv`:

    uv sync                                   # create the environment (once)
    uv run python step1_build_sessions.py     # step 1, about 2.5 minutes
    uv run python step2_preprocess.py         # step 2, about 30 seconds
    caffeinate -i uv run python step3_fit_rrr.py   # step 3, about 1 hour

`caffeinate -i` keeps the Mac awake during the long run. Step 3 saves each fit
as soon as it is done, so if it stops, running it again resumes where it left
off (delete `outputs/rrr/` to start over).

`pyproject.toml` was created with these commands (`uv.lock` and `.venv/` are
generated from it automatically):

    uv init --bare --python 3.12
    uv add "brainwidemap @ git+https://github.com/int-brain-lab/paper-brain-wide-map.git@de37454"
    uv add --editable ../ibl-ai-agent
    uv add numpy pandas pyarrow scipy matplotlib scikit-learn torch
    uv add tqdm

`ibl-ai-agent` is only used to read the agent's compressed data files.
`brainwidemap` is needed because the agent depends on it, and it is only on GitHub.

## Files

- `step1_build_sessions.py`, `step2_preprocess.py`, `step3_fit_rrr.py`: steps 1-3.
- `data/`: the two lists copied from the authors (see Inputs).
- `outputs/sessions/`: one file per session, made by step 1, used by step 2.
- `outputs/preprocessed/`: one file per session, made by step 2, used by step 3
  (2.8 GB).
- `outputs/rrr/`: the fitted model and the per-neuron test R², made by step 3,
  used by step 4.
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
- `dt`, `neural_start`: the time axis (0.01 s bins, starting 0.2 s before the
  stimulus). Later steps read it from here instead of redefining it.

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

## Step 2: preprocessing

`step2_preprocess.py` turns each session from step 1 into what the model is
fitted on:

- Targets, y (`prepare_targets`), as in the paper's Methods ("Target
  variables", eq. 1): spike counts smoothed in time with a Gaussian (s.d.
  20 ms), then z-scored per neuron and time bin across trials. Smoothing turns
  noisy 0/1 spike counts into an estimate of the firing rate. Z-scoring puts
  every neuron and time bin on the same scale, so the fit weighs them equally
  and coefficients can be compared across neurons. Because the mean across
  trials is removed at every time bin, y only contains how each trial differs
  from the average trial; that is what the model explains.
- Inputs, X (`prepare_inputs`): the 8 variables at every trial and time bin.
  - Task variables, constant within a trial, encoded as in the paper's Methods
    ("Input variables"): block (+1 = left block, p(left) = 0.8), side (+1 =
    stimulus on the left), contrast (0 for 0%, 1 for low contrast up to 12.5%,
    4 for high), choice (+1 = clockwise), outcome (+1 = reward). The authors'
    code computes the outcome as "stimulus side matches the choice"; we take
    IBL's feedbackType, which is identical in all 60,528 trials.
  - Movement variables (wheel, whisker, lick): neurons and movements are not
    perfectly in sync, so each trace is shifted in time to where it lines up
    best with the session's neurons; one shift per variable and session. The
    paper's Methods only give the idea ("shifted the movement traces so that
    the zero-lagged correlation was maximized", "averaged across neurons and
    trials"). The details come from the authors' code (`find_bestdelay_byCC`,
    lines 227-247, and `_preprocess_movement`, line 89, in
    `utils/save_and_load_data.py`), and we copy them:
    - the shift ranges over +-100 ms, which is what their -0.3 to 0.9 s
      behaviour window allows;
    - the measure is a cross-correlation (their "CC"): for each shift, each
      neuron's dot product with the shifted trace (after removing each trial's
      mean, summed over time, averaged over trials), with the neurons combined
      as the square root of the sum of squares. It is not normalized (values
      are not between -1 and 1), and the neurons are not plainly averaged as
      the Methods' wording suggests: with a plain average, neurons with
      opposite signs would cancel;
    - if the best shift is at the edge of the range (+-100 ms), the search
      counts as failed and the trace is used unshifted.
  - Every variable is then z-scored per time bin across trials, as in the
    paper's Methods (so the coefficients are comparable across variables, and
    the ridge penalty treats them equally). The authors' code also z-scores
    each movement trace over all bins first; the per-bin z-score undoes that,
    so we skip it.

Each session is saved to `outputs/preprocessed/<eid>.npz`:

- `X`: (trials, 100 bins, 8): the 8 variables in Fig. 2 order (block, side,
  contrast, choice, outcome, wheel, whisker, lick).
- `y`: (trials, 100 bins, neurons).
- `y_mean`, `y_std`: (100 bins, neurons), to turn predictions back into
  (smoothed) spike counts.
- `variables`, `area`, `pid`, `cluster_id`, `dt`, `neural_start`: names of
  X's columns, the neuron table and the time axis, passed on from step 1.

Results: all 170 sessions are preprocessed, in about 30 seconds. The chosen
shifts are in `outputs/checks/step2_summary.csv` and
`outputs/checks/step2_check.png`. They make sense: neurons tend to fire before
the wheel moves (median +50 ms, motor commands) and after the whiskers move
(median -30 ms, sensing the movement). The shift search fails in 37 sessions
for the wheel, 27 for whisker and 19 for lick, and these failures continue the
same trends: almost all wheel failures want more than +100 ms, all whisker
failures less than -100 ms. So the +-100 ms range the authors' code allows is
too narrow for some sessions. We keep it and the failure rule, to match them;
neither is mentioned in the paper (see Limitations of the method).

How it was checked: on 5 sessions, we computed X and y with the authors' code
(their actual `find_bestdelay_byCC`, and the rest of `_read_Xy` rewritten line
by line) and compared. y is identical, X differs by at most 3e-7 (rounding),
and every chosen shift is the same.

## Step 3: fitting the RRR model

`step3_fit_rrr.py` fits the paper's encoding model to all neurons of all
sessions at once.

- The model (`RRR`; paper Methods, eq. 2-3): each neuron's z-scored activity
  is a linear combination of the 8 variables, with time-varying coefficients,
  plus a baseline per neuron and time bin:
  `y_hat[n](k, t) = sum_v beta[n, v](t) * x[v](k, t) + b[n](t)`. Every
  coefficient's time course is a mix of a few time shapes shared by all
  neurons and sessions: `beta[n, v](t) = sum_i U[n, v, i] * V[i](t)`. This
  reduced rank is what makes the fit possible with few trials: 8 x 5 = 40
  numbers per neuron instead of 8 x 100, and every neuron helps estimate V.
  This is why all brain areas are fitted, not only cortex.
- The fit (`fit`; paper Methods, eq. 7): minimize the squared prediction error
  plus a ridge penalty, lambda times the sum of all beta² (the baseline is not
  penalized), with L-BFGS and the authors' settings, over all sessions together.
- No hyperparameter search: we fix d = 5 shapes (the paper's best) and lambda =
  75 (see Differences).
- It is fitted twice over:
  - 3 times on 70% of each session's trials, measuring each neuron's R² on the
    other 30%, trials the model never saw (`test_r2`). Averaged over the 3
    splits, this is the honest goodness of fit that step 4 uses to keep the
    well-fit neurons (paper Methods: "We repeated the whole
    split-train-test process three times"). The test fraction is 0.3 as in
    the authors' code; the paper says one-third.
  - once on all trials. Its coefficients are what Fig. 2c summarizes.
- The splits are stratified by block x contrast, so every combination is in
  both parts (paper Methods); if a session has too few trials of some
  combination, by block only, then not at all (authors' code). The paper
  doesn't say why only these two variables. Our reading: they are set by the
  experimenter rather than by the mouse's behaviour, they are the ones most
  likely to be unbalanced by chance (blocks come in long runs; 0% contrast is
  ~10% of trials), and adding more variables would leave too few trials per
  combination to split.
- R² is computed on spike counts: the data and the predictions are
  un-z-scored first (authors' code). The two versions differ. Per neuron,
  with s(t) and m(t) the std and mean across trials at time bin t, the errors
  in counts are the z-scored errors weighted by s(t)², and the total variation
  in counts is K * sum_t s(t)² (trial-to-trial) + K * sum_t (m(t) - mean)² (the
  average response over time). So on spike counts the average response also
  counts as explained, and R² is higher than on z-scored data. We use spike
  counts because the paper's goodness of fit and its threshold on R² minus the
  average-response model's R² (step 4) are defined on firing rates.

Output, in `outputs/rrr/`:

- `split0.npz`, `split1.npz`, `split2.npz`: for each session id, one test R²
  per neuron (same order as in step 2's files); `losses`, the fit's loss after
  every evaluation.
- `all_trials.npz`: the final model. `V` (5 shapes x 100 bins); for each
  session `U_<eid>` (neurons x 8 variables x 5) and `b_<eid>` (neurons x 100
  bins); `losses`. The coefficients are beta = U @ V.

Results: about 1 hour on a 16 GB M3 laptop. Every fit converged long before
the 1000-iteration limit (205 to 358 evaluations). The 5 learned shapes
(`outputs/checks/step3_check.png`) are smooth and follow the task's timeline:
flat before the stimulus, with features around the stimulus response
(~0.1 s), the movement (~0.2-0.3 s) and after the response (~0.4 s and
later). Individual shapes are not unique (other sets of 5 give the same
coefficients), so only the coefficients are interpreted. The per-neuron test R²
has a median of 0.049 and a long tail up to ~0.7; it includes the part the
average response explains, which step 4 subtracts.

How it was checked: a first fit on all trials, inspected before the full run
(loss converging, smooth shapes); a test run of the whole script on 3 sessions
with 10 iterations, including the saving and the resume. The model, the loss
and the way initial values are drawn are the authors' (`RRRGD_model`); the
random initial values themselves differ, which doesn't matter once the fit
has converged.

## Limitations of the method

These come from the authors' method, which we copy on purpose. They are
open questions, not differences from the authors.

- The model is instantaneous: a neuron's activity at time t is explained by
  the variables at the same time t. A neuron that fires 50 ms before the wheel
  moves is only captured because the wheel trace is shifted first (step 2).
  The paper acknowledges that their model, unlike GLMs, can't use inputs from
  neighbouring time steps, and says it "could be incorporated in future
  extensions" (Methods, "Comparison to previous regression models").
- One shift per movement and session, shared by all of the session's neurons.
  A session usually mixes neurons that lead a movement (motor-like) and
  neurons that lag it (sensory-like). The shift is computed from all neurons
  together, so it goes to whichever group contributes more, or lands between
  them. The other neurons get a slightly mistimed trace and their selectivity
  to that movement is underestimated (weakened, not lost: movements last a few
  hundred ms and the spikes are smoothed). A probe crosses several areas, so
  this can bias the comparison between areas. The paper doesn't discuss it.
  Possible test: compute a best shift per neuron and see how much they spread
  within a session.
- The shift range is +-100 ms, and many sessions want more (see step 2
  results). Possible test: widen the range (needs a wider behaviour window in
  step 1) and see whether Fig. 2c changes.

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
- No hyperparameter search: the authors choose d (number of time shapes) from
  3-6 and lambda (ridge penalty) from 25, 75 and 200 by cross-validation. We
  fix d = 5, the best value reported in the paper, and lambda = 75, the middle
  of their grid (the paper doesn't report which lambda won). lambda shrinks all
  coefficients; Fig. 2c z-scores each variable across areas, so a shrinkage
  that affects all areas alike mostly cancels out, but it can change which
  neurons pass step 4's threshold. Testing 25 and 200 is a possible
  sensitivity check.

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
- Step 2 (smoothing, z-scoring, task variable encoding): `_read_Xy` in
  `utils/save_and_load_data.py`. Movement shifts: `find_bestdelay_byCC`
  (lines 227-247, including the failure rule at 242-246) and
  `_preprocess_movement` (line 89: failed search -> unshifted) in the same file.
  `_read_Xy` also appends a column of ones to X, so that their model's
  baseline (`RRR_b`) is the coefficient of that column. We don't: our model
  adds the baseline explicitly. Same model, same results.
- Step 3: the model, its initial values, the loss and the ridge penalty are
  `RRRGD_model` in `utils/RRRGD.py`. The L-BFGS fit and the cross-validation
  (70/30 splits, seeds 42 + split, R² on held-out trials) are
  `train_model_hyper_selection` in `utils/RRRGD_main_CV.py`, called with the
  grid and the stratification order (block x contrast, block, none) from
  `step2_train_RRR.py`. R² per neuron, un-z-scored and clipped at 0:
  `compute_R2_RRRGD` in `utils/RRRGD.py` and `compute_R2_main` in
  `utils/utils.py`. Their fit sums all sessions' losses and computes the
  gradient once; we compute it session by session and add the gradients
  (gradient accumulation), so that the fit fits in 16 GB of memory. Same
  gradients, same fit. After choosing the hyperparameters they refit the 3
  splits under new file names; with fixed hyperparameters we need each split
  only once.
