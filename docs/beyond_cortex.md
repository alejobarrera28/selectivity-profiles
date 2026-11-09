# Beyond cortex: this research's decisions

The validation is done: on the `shuqi` dataset, our implementation reproduces
Posani-Wang's trained model ([pipeline.md](pipeline.md), Validation against
Posani-Wang's model). From here on, the work is our own: the same engine
applied to the whole brain, starting with the `bwm` dataset. This file
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
  cross them. Done with IBL's 2023_12 release of the Brain Wide Map
  (`data/bwm_2023_12_release.csv`, provenance in [sources.md](sources.md)),
  see "The session list" below.
- Neurons: IBL's good units only (the ones that pass all its quality
  metrics), instead of every unit as in Posani-Wang.
- Data version: IBL's data as of a fixed date (`as_of` in `run.py`,
  2026-11-01), which can be moved later.
- First analyses: decoding and basic clustering.
- The 11 sessions that differ from Posani-Wang (differences.md, Open
  questions): report what we found to Shuqi Wang, and ask Olivier at IBL if
  it stays unresolved. They don't block the work.

## The session list (2026-11-08)

The `bwm` dataset reads the probes of IBL's 2023_12 Brain Wide Map release:
699 probes in 459 sessions. In each session it reads only the listed probes
(`probes` in `run.py`); `shuqi` reads all the probes of its sessions.

What the list is. IBL's `bwm_query` selects the probes of the Brain Wide Map
project (`ibl_neuropixel_brainwide_01`) whose session is not a mock, is not
session QC CRITICAL (`session__qc__lt,50`; FAIL is allowed), and passes IBL's behaviour criteria (`extended_qc__behavior = 1`); whose probe
is not QC CRITICAL, has its histology traced, and has its anatomical
alignment resolved; and whose session passes 10 task QC checks in at least 90%
of its trials, or was marked PASS by the experimenter. Run with
`freeze="2023_12_bwm_release"` (the default), it returns the CSV of its
release instead of querying, which is what the Brain Wide Map papers use.

Why a fixed list. The server can be queried directly (`bwm_query` with
`freeze=None` does it, paginated by Alyx's REST API), so a list is not needed
to reach the data. But the QC fields it filters on are the database's current
state, not versioned like the files (ONE's `as_of` doesn't apply to them), so
the same query gives a different set as IBL revises its QC. Run on
2026-11-08 it gave 694 probes in 456 sessions: 6 of the release's probes are
gone (5 Witten-lab probes of 2020-09-02/03, whose sessions are now QC FAIL on
the task checks, and one Angelaki-lab probe of session 288bfbf3, now probe QC
CRITICAL) and one is new (a Hofer-lab probe, 19baa84c). The release is fixed,
official and comparable with IBL's own analyses, and matches the fixed
cutoff decided with Liam.

Checks (2026-11-08): the copy is the same file as IBL's repo (same SHA-1,
433162ec, on its main branch and at the installed commit); every one of its
459 sessions is on the public server; 458 of them have exactly the listed
probes, and the other (dfd8e7df) has one more, f936a701, which is probe QC
CRITICAL, so leaving it out is right. Against Posani-Wang: 172 of their 178
sessions are in it, and in those, every probe they read (all the session's
probes, `eid2pid` in their `utils/download_data.py`) is listed, so on their
sessions the list is a superset. Their other 6 (30c4e2ab, 754b74d5,
b03fbc44, c51f34d8, d9f0c293, f140a2ec) are Brain Wide Map project sessions
in none of IBL's releases (2022_10 or 2023_12), all session QC FAIL today
(c51f34d8's probe also CRITICAL). How they built their list isn't in their
repo; their Methods select sessions by cortical targets. Loading one session with its listed probe and good units only works
(45 s). To watch on the first run: the probe now CRITICAL (7dcd74a9, session
288bfbf3) may not be served by ONE, which hides files flagged CRITICAL; if
so, the session shows up as "no data" in the sessions summary.

The list fixes which probes; their files are read as of `as_of`
(2026-11-01), so with IBL's latest revisions up to then. IBL's own Brain Wide
Map code (`bwm_loading.py` on its main branch) instead pins older ones: the
spike sorting at revision 2024-05-06 and the trials before 2024-07-10
(`MODIFIED_BEFORE`), so its papers' data lack the later corrections that
`bwm` has. Its later "releases" (tags 2024_Q2 and 2026_Q2) are aggregate
tables of trials and units, not new session lists: the list stays 2023_12.

What the list leaves out (2026-11-08), to judge whether it should be the
method. On the public server, the Brain Wide Map project has 742 probes in 501
sessions; the release keeps 699 in 459. The other 43 probes, each in a
different session (mostly 2020-2021), all fail at least one of the query's
criteria: 30 IBL's behaviour criteria (`extended_qc.behavior`, the mouse's
performance in that session), 14 the task QC checks, 12 probe QC CRITICAL, 12
alignment not resolved (so their neurons' areas aren't reliable), 5 session
QC CRITICAL; one (19baa84c) passes today, after the release. Posani-Wang's 6
sessions outside the list are among them: 5 only for the behaviour criteria,
c51f34d8 for its probe CRITICAL. Outside the project, the server has other
IBL experiments with probes (1,393 sessions in all), different on purpose:
`angelaki_mouseASD` (155 sessions, the same task in mouse models of autism),
the Witten lab's dopamine photometry (`witten_learning_dop`, `witten_dop`,
over 500, another task), `churchland_learninglifespan` (28, learning and
age), `serotonin_inference` (46, optogenetic stimulation); they are not
missing sessions, and adding them would mix populations. Conclusion: nothing
is left out by accident; each exclusion is an explicit QC criterion. The one
open to discussion is the behaviour criteria: a session where the mouse
performed below IBL's criterion can still have more than 100 valid trials and
good neural data, and Posani-Wang included some. Open decision, not raised
with Liam yet; adding those sessions later only reads the new ones.

Alternatives, not taken: the query run today and saved by code as a new list
(IBL's latest QC, but not comparable with its papers and drifting with each
rerun); or criteria of our own (e.g. every probe of the project with our own
QC).

## Kept from Posani-Wang for now

Liam wasn't asked about these. We keep Posani-Wang's choice, and revisit it if
the check shows a problem on the `bwm` data.

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
