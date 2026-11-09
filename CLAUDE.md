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

The implementation has been validated against Posani-Wang's trained model (the
`posani_wang` dataset). The work now is our own research on the whole brain (the
`bwm` dataset): its decisions, and the Posani-Wang choices still to be
checked on the new data, are in docs/beyond_cortex.md.

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
  <dataset>`) that read the checkpoints and write to a folder of their own,
  `outputs/<dataset>/results/<name>/`. The check figure of each stage
  (`analyses/checks.py`) is drawn by `run.py` as the stage ends, and goes with
  the stages' summary tables to `outputs/<dataset>/results/pipeline/`.
- Every file in the repo is either written by hand on purpose or produced by
  the code. No files from hand-run commands. Of `outputs/`, only each dataset's
  `results/` (figures and tables) is committed.
- Module layout: module docstring (what it does); constants at the top,
  grouped by topic (`# ---- Trials`, `# ---- Model`, ...), each with its source
  ("Posani-Wang Methods", "Posani-Wang code", or "ours" with its reason);
  then `# =====` sections per entity, helpers prefixed with `_`, one public
  function per entity (`prepare_*`, `fit`, ...). `run.py` has no `# =====` sections: its functions follow the
  pipeline's order. Mains and stage functions use numbered comments
  (`# 1. ...`) for their steps.
- Rules that select trials or neurons are named entries of a `rules`
  dictionary, each tagged `# Posani-Wang Methods`, `# Posani-Wang code` or
  `# ours`.
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
- docs/beyond_cortex.md: this research's own decisions (what, why, with whom,
  when), and the Posani-Wang choices kept for now, each with how to check it.
  New decisions go here.
- docs/differences.md: the record of the validation: differences from
  Posani-Wang, limitations of their method (still valid), open questions.
- docs/sources.md: inputs' provenance and where each rule lives in the
  Posani-Wang code.

Any change to a stage or an analysis updates the matching docs. Plain,
readable prose; no heavy formatting.

## Workflow

- Explain a change (what, how, why, its source) before or right after making
  it; walk through new code before committing.
- What follows Posani-Wang is validated against their code where possible,
  and the result recorded briefly in docs/pipeline.md. A choice of our own is
  recorded with its reason in docs/beyond_cortex.md, and its effect checked
  on the data.
- A reorganization moves code verbatim: same names, docstrings, comments and
  prints; only what the move requires changes, and the diff is reviewed.
- Never commit unreviewed code. One commit per change; ask before pushing.
