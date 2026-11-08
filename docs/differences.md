# Differences, limitations and open questions

Everything that is not a plain copy of the Posani-Wang paper or code: where
we differ from them and why, the limitations of the method we copy on purpose,
and what to compare our results against.

## Differences from Posani-Wang

- Data: the `shuqi` dataset reads IBL's data as Posani-Wang had it, so it
  differs from them only in the method, except for 18 sessions excluded by
  hand ([sources.md](sources.md)) and 11 whose data differ (Open questions).
  `mine` differs in two ways:
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
  after it). This drops 3 of their sessions.
- Wheel velocity: the Posani-Wang code filters the wheel position trial by trial; we
  filter the whole session at once with the same filter, which avoids edge
  effects at the start and end of each trial. Small difference.
- No hyperparameter search: Posani-Wang chooses d (number of time shapes) from
  3-6 and lambda (ridge penalty) from 25, 75 and 200 by cross-validation, as
  the values with the highest mean test R². We fix d = 5, the best value
  reported in Posani-Wang, and lambda = 25 (Posani-Wang doesn't report which
  lambda won). On the `shuqi` dataset, lambda = 25 gives a higher test R² than
  75 (mean 0.071 against 0.059), so their criterion picks it over 75, and it
  reproduces their trained model: mean test R² 0.071 against their 0.072, and
  93% agreement on which neurons pass the profiles' threshold, against 80% with
  lambda = 75. lambda shrinks all coefficients: with 25 our alpha come out
  11-24% larger than theirs depending on the variable (16% on average; with 75,
  4-21% smaller), with the same shape (median cosine similarity 0.996 per
  neuron, 0.994 per session and variable). This scale is common to all areas,
  and the area selectivity's z-score removes it exactly. Shuqi Wang (Slack,
  2026-11-05): only the relative magnitudes of alpha matter, not their scale.

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
sessions stage); lambda = 75 and 200; subtracting each area's alpha with the variable
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
  so the `shuqi` dataset can be compared with it neuron by neuron
  (`analyses/compare_posani_wang.py`; results in [pipeline.md](pipeline.md),
  Validation against Posani-Wang's model).

## Open questions

- 11 of Posani-Wang's sessions don't match ours in the `shuqi` dataset (the
  red sessions of `analyses/compare_posani_wang.py`): 28338153, 29a6def1,
  2d9bfc10, 3bcb81b4, 5c0c560e, 642c97ea, 7cc74598, caa5dddc, ebce500b,
  f4eb56a4, f56194bc. What is known:
  - They are exactly the sessions of their list recorded in 2023: 10 from the
    Steinmetz lab (mice NR_0024, NR_0028, NR_0031, Jan-Jul 2023) and 1 from
    the Churchland lab at UCLA (MFD_09, Oct 2023). The 11 Steinmetz sessions of
    2022, with the same probe type (3B2), match exactly.
  - We have 3-8 times more neurons there (3,494 that they don't have), and
    they have 348 that we don't. The neurons both have give different
    results, even the null model's R², which has no fit: in their version the
    activity of each trial differs from ours, so either the trial times or
    the spike times of those sessions differ. Without their data it can't be
    told which.
  - Often a neuron's R² is near 0 in one version and clearly above in the
    other, both ways (null model: 40% of the 370 with R² > 0.1 in either;
    never in the other sessions). Their profiles agree with ours at chance
    level (cosine similarity 0.785 per session and variable, 0.78 with the
    neurons shuffled; signed beta over time, 0.04).
  - In our version they look right: the wheel separates by choice after the
    stimulus in all 11, and the population responds to the stimulus in 8
    (weakly in 28338153, 7cc74598 and ebce500b). The trials' revision of
    2024-07-15 doesn't change them.
  Sent to Shuqi Wang (2026-11-05); Liam Paninski suggests asking Olivier at
  IBL if it stays unresolved.
- Fig. 2b: Posani-Wang reports a mean R² of 0.16 for the RRR and 0.09 for the
  "PSTH per task condition". Their own trained model gives, in cortex, 0.068
  for the RRR and 0.050 for the null model (ours on the `shuqi` dataset: 0.067
  and 0.049), so the gap is not in our implementation. Fig. 2b's baseline is an
  average per task condition (Posani-Wang doesn't define the conditions), a
  stronger baseline than the null model of the Methods and the Posani-Wang
  code (one average over all trials), which is the one the selection uses; why
  their published RRR R² is higher than their trained model's is unknown.
- Why the cross-validation splits are stratified only by block and contrast
  (stated in Posani-Wang, not justified; see [pipeline.md](pipeline.md), the model).
- Whether the lead/lag compromise of the single shift per session explains
  our whisker differences in the area selectivity figure.
