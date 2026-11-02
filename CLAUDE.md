# Working on this repo

Context for anyone (developer or coding agent) continuing this project. Start
with README.md, then docs/.

## What this is

A research project, in progress, on the selectivity profiles of single
neurons across the whole mouse brain, on the IBL Brain Wide Map. The encoding
model follows Posani, Wang et al. (Nature 2026; "Posani-Wang" in the code and
docs), reimplemented from scratch;
their code (https://github.com/realwsq/brainwide-RRR-encoding-model, commit
00d40b9) is the reference for the method's details. The data is read from
IBL's public server (`pipeline/ibl.py`). Understanding every step
matters more than speed: each choice must be explainable and traceable to the
Posani-Wang paper or code, or to a recorded decision of ours.

## Conventions

- The main pipeline is `run.py <dataset>`: four stages (sessions, inputs,
  model, profiles), each with its module in `pipeline/` and its checkpoint in
  `outputs/<dataset>/`. Each stage reads the previous checkpoint; a stage whose
  checkpoint exists is skipped.
- Datasets are named entries of `DATASETS` in `run.py`, holding the data
  parameters (where the data comes from, which sessions). The method's
  constants stay in the modules.
- Modules in `pipeline/` are functions only: no figures, and no files read or
  written, except `pipeline/ibl.py`, which reads the raw IBL data. All reading
  and writing of checkpoints is in `run.py`.
- Figures and analyses are scripts in `analyses/` (`python -m analyses.<name>
  <dataset>`) that read the checkpoints and write to `outputs/<dataset>/results/`.
- Every file in the repo is either written by hand on purpose or produced by
  the code. No files from hand-run commands. Of `outputs/`, only each dataset's
  `results/` (figures and tables) is committed.
- Module layout: module docstring (what it does); constants at the top,
  grouped by topic (`# ---- Trials`, `# ---- Model`, ...), each with its source
  ("Posani-Wang Methods", "Posani-Wang code"); then `# =====` sections per
  entity, helpers prefixed with `_`, one public function per entity (`prepare_*`,
  `fit`, ...). `run.py` has no `# =====` sections: its functions follow the
  pipeline's order. Mains and stage functions use numbered comments
  (`# 1. ...`) for their steps.
- Rules that select trials or neurons are named entries of a `rules`
  dictionary, each tagged `# Posani-Wang Methods` or `# Posani-Wang code`.
- Values decided in one stage travel with the data (e.g. the time axis `dt`,
  `neural_start`) instead of being redefined in later stages or analyses.
- Comments explain decisions and sources, not mechanics; no comments about
  non-issues; no special cases just to silence harmless warnings. Docstrings
  say what goes in and out, the method, and where it comes from; the longer
  reasoning goes in docs/.
- Keep the code simple and the repo minimal: add a file or a dependency only
  when a stage or an analysis needs it.
- Plain `print` for progress; `tqdm` for long fits.

## Documentation

- README.md: what the project is, how the code is organized, how to run it.
- docs/pipeline.md: per stage: what and why, outputs, results, validation.
- docs/differences.md: differences from Posani-Wang, limitations, open questions.
- docs/sources.md: inputs' provenance and where each rule lives in the
  Posani-Wang code.

Any change to a stage or an analysis updates the matching docs. Plain,
readable prose; no heavy formatting.

## Workflow

- Explain a change (what, how, why, its source) before or right after making
  it; walk through new code before committing.
- Validate each stage against the Posani-Wang code where possible, and record
  the result briefly in docs/pipeline.md.
- A reorganization moves code verbatim: same names, docstrings, comments and
  prints; only what the move requires changes, and the diff is reviewed.
- Never commit unreviewed code. One commit per change; ask before pushing.
