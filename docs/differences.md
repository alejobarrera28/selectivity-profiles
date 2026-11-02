# Differences, limitations and open questions

Everything that is not a plain copy of the Posani-Wang paper or code: where
we differ from them and why, the limitations of the method we copy on purpose,
and what to compare our results against.

## Differences from Posani-Wang

- Data: the `shuqi` dataset reads IBL's data as Posani-Wang had it, so it
  differs from them only in the method. `mine` differs in two ways:
  - Neurons: only IBL's "good units" (the ones that pass all its quality
    metrics), while Posani-Wang used every unit firing between 0.5 and 50 Hz.
    In their sessions that's about 5,800 cortical candidates instead of
    39,000, so fewer areas reach the 30 well-fit neurons needed to appear in
    the area selectivity figure.
  - Data version: IBL's current data. In 2024 IBL re-ran, on every recording,
    the processing that extracts the neurons ("2024-05-15 Spike-sorting
    re-run" in the changelog of IBL's Brain Wide Map release page): in
    Posani-Wang's sessions it found about 1.7 times as many neurons, and gave
    every neuron a new id, so the two versions share no neuron. IBL also
    revised the trials, licks and motion energy of most sessions in 2025.
- Minimum of 100 trials per session: we count the trials left after all
  selection rules. The Posani-Wang code counts before removing the 50/50 block,
  which looks like an accident (the model is only fitted on the trials left
  after it). This drops 2 sessions they would keep (about 20 cortical neurons).
- Wheel velocity: the Posani-Wang code filters the wheel position trial by trial; we
  filter the whole session at once with the same filter, which avoids edge
  effects at the start and end of each trial. Small difference.
- No hyperparameter search: Posani-Wang chooses d (number of time shapes) from
  3-6 and lambda (ridge penalty) from 25, 75 and 200 by cross-validation. We
  fix d = 5, the best value reported in Posani-Wang, and lambda = 75, the middle
  of their grid (Posani-Wang doesn't report which lambda won). lambda shrinks all
  coefficients; the area selectivity z-scores each variable across areas, so a shrinkage
  that affects all areas alike mostly cancels out, but it can change which
  neurons pass the profiles' threshold.

Implementation differences that don't change results are listed in
[sources.md](sources.md) (the baseline added explicitly instead of through a
column of ones, gradient accumulation, the outcome taken from IBL's
feedbackType).

## Limitations of the method

These come from Posani-Wang's method, which we copy on purpose.

- The model is instantaneous: a neuron's activity at time t is explained by
  the variables at the same time t. A neuron that fires 50 ms before the wheel
  moves is only captured because the wheel trace is shifted first (inputs).
  Posani-Wang acknowledges that its model, unlike GLMs, can't use inputs from
  neighbouring time steps, and says it "could be incorporated in future
  extensions" (Methods, "Comparison to previous regression models").
- One shift per movement and session, shared by all of the session's neurons.
  A session usually mixes neurons that lead a movement (motor-like) and
  neurons that lag it (sensory-like). The shift is computed from all neurons
  together, so it goes to whichever group contributes more, or lands between
  them. The other neurons get a slightly mistimed trace and their selectivity
  to that movement is underestimated (weakened, not lost: movements last a few
  hundred ms and the spikes are smoothed). A probe crosses several areas, so
  this can bias the comparison between areas. Posani-Wang doesn't discuss it.
- The shift range is +-100 ms (set by Posani-Wang's behaviour window, not
  mentioned in Posani-Wang), and many sessions want more (inputs results).
- The selectivity alpha = sum over time of |beta(t)| has a noise floor:
  coefficients estimated from noisy data are never exactly 0, so even a
  neuron that doesn't care about a variable gets a positive alpha. Z-scoring
  across areas removes a floor that is the same everywhere, but areas whose
  coefficients are noisier (fewer trials or noisier neurons) get a higher
  floor and can look slightly more selective. Neither Posani-Wang nor the
  Posani-Wang code corrects for it.

Possible sensitivity tests: a best shift per neuron (how much they spread
within a session); a wider shift range (needs a wider behaviour window in the
sessions stage); lambda = 25 and 200; subtracting each area's alpha with the variable
shuffled across trials.

## What to compare our results against

- Posani-Wang: about 14,000 cortical neurons after selection, of which 4,617
  are fitted well enough by the model (delta R² above 0.015). Fig. 2c shows
  31 areas.
- Posani-Wang's repo has its own Fig. 2c output
  (`Fig2_globally_structured_brain/abs_sel_per_area.pdf`), made from their
  pre-trained model. It doesn't exactly match the published figure: it has
  VISC instead of FRP, and a few of the top-3 markers differ (e.g. for "side").
  Their README says IBL data changes across versions and offers the exact data
  on request (shuqi.wang@epfl.ch). So an exact match isn't expected; the
  overall pattern is what to compare.
- Their pre-trained model (`trained_RRR_model/RRR_selectivity.json`, 1.27 GB,
  stored with Git LFS) has each neuron's IBL id, coefficients, R² and null R²,
  so the `shuqi` dataset can be compared with it neuron by neuron.

## Open questions

- Fig. 2b: Posani-Wang reports a mean R² of 0.16 for the RRR and 0.09 for the
  "PSTH per task condition". Ours, in cortex: 0.093 for the RRR and 0.068 for
  the null model. Two things differ. Fig. 2b's baseline is an average per task
  condition (Posani-Wang doesn't define the conditions), a stronger baseline
  than the null model of the Methods and the Posani-Wang code (one average over
  all trials), which is the one the selection uses. And our RRR's R² is
  clearly lower; the cause is unknown (good units only, lambda not searched,
  or a different set of neurons in Posani-Wang's average). Comparing the
  `shuqi` dataset with the per-neuron `RRR_r2` and `null_r2` in Posani-Wang's
  pre-trained model would tell.
- Why the cross-validation splits are stratified only by block and contrast
  (stated in Posani-Wang, not justified; see [pipeline.md](pipeline.md), the model).
- Whether the lead/lag compromise of the single shift per session explains
  our whisker differences in the area selectivity figure.
