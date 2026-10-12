# Differences, limitations and open questions

Everything that is not a plain copy of the paper or the authors' code: where
we differ from them and why, the limitations of the method we copy on purpose,
and what to compare our results against.

## Differences from the authors

- Neurons: we only have "good units" (IBL's best-quality neurons), while the
  authors used every neuron firing between 0.5 and 50 Hz. In the authors'
  sessions that's about 5,800 cortical candidates instead of 39,000; after
  step 1 we have 4,233 cortical neurons, while the paper reports about 14,000.
  So fewer areas reach the 30 well-fit neurons needed to appear in the area
  selectivity figure (16 instead of 31). This is by far the largest difference.
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
  coefficients; step 5 z-scores each variable across areas, so a shrinkage
  that affects all areas alike mostly cancels out, but it can change which
  neurons pass step 4's threshold.

Implementation differences that don't change results are listed in
[sources.md](sources.md) (the baseline added explicitly instead of through a
column of ones, gradient accumulation, the outcome taken from IBL's
feedbackType).

## Limitations of the method

These come from the authors' method, which we copy on purpose.

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
- The shift range is +-100 ms (set by the authors' behaviour window, not
  mentioned in the paper), and many sessions want more (step 2 results).
- The selectivity alpha = sum over time of |beta(t)| has a noise floor:
  coefficients estimated from noisy data are never exactly 0, so even a
  neuron that doesn't care about a variable gets a positive alpha. Z-scoring
  across areas removes a floor that is the same everywhere, but areas whose
  coefficients are noisier (fewer trials or noisier neurons) get a higher
  floor and can look slightly more selective. Neither the paper nor the
  authors' code corrects for it.

Possible sensitivity tests: a best shift per neuron (how much they spread
within a session); a wider shift range (needs a wider behaviour window in step
1); lambda = 25 and 200; subtracting each area's alpha with the variable
shuffled across trials.

## What to compare our results against

- The paper: about 14,000 cortical neurons after selection, of which 4,617
  are fitted well enough by the model (delta R² above 0.015). Fig. 2c shows
  31 areas.
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

## Open questions

- Fig. 2b: the paper reports a mean R² of 0.16 for the RRR and 0.09 for the
  "PSTH per task condition". Ours, in cortex: 0.093 for the RRR and 0.068 for
  the null model. Two things differ. Fig. 2b's baseline is an average per task
  condition (the paper doesn't define the conditions), a stronger baseline
  than the null model of the Methods and the authors' code (one average over
  all trials), which is the one the selection uses. And our RRR's R² is
  clearly lower; the cause is unknown (good units only, lambda not searched,
  or a different set of neurons in the paper's average). Comparing with the
  per-neuron `RRR_r2` and `null_r2` in the authors' pre-trained model would
  tell.
- Why the cross-validation splits are stratified only by block and contrast
  (stated in the paper, not justified; see [pipeline.md](pipeline.md), step 3).
- Whether the lead/lag compromise of the single shift per session explains
  our whisker differences in the area selectivity figure.
