# Working on this repo

Context for anyone (developer or coding agent) continuing this analysis. Start
with README.md, then docs/.

## What this is

A from-scratch reimplementation of the RRR encoding-model analysis of Posani,
Wang et al. (Nature 2026), on IBL Brain Wide Map data downloaded by the IBL AI
agent (`../ibl-ai-agent`). The authors' code, for reference, is
https://github.com/realwsq/brainwide-RRR-encoding-model (commit 00d40b9).
Understanding every step matters more than speed: each choice must be
explainable and traceable to the paper, the authors' code, or a recorded
decision of ours.

## Conventions

- One script per step (`stepN_*.py`), run in order. Each reads the previous
  step's files and writes its own data to `outputs/`, plus its figures and
  tables to `outputs/results/`.
- Every file in the repo is either written by hand on purpose or produced by
  the code. No files from hand-run commands. Of `outputs/`, only `outputs/results/`
  (figures and tables) is committed.
- Script layout: module docstring (what the step does, its output); constants
  at the top, grouped by topic (`# ---- Trials`, `# ---- Model`, ...), each
  with its source ("paper Methods", "authors' code"); then `# =====` sections
  per entity, helpers prefixed with `_`, one public function per entity
  (`prepare_*`, `fit`, ...); all data reading in the `__main__` block.
- Rules that select trials or neurons are named entries of a `rules`
  dictionary, each tagged `# paper Methods` or `# authors' code`.
- Values decided in one step travel with the data (e.g. the time axis `dt`,
  `neural_start`) instead of being redefined in later steps.
- Comments explain decisions and sources, not mechanics; no comments about
  non-issues; no special cases just to silence harmless warnings. Docstrings
  say what goes in and out, the method, and where it comes from; the longer
  reasoning goes in docs/.
- Keep the code simple and the repo minimal: add a file or a dependency only
  when a step needs it.
- Plain `print` for progress; `tqdm` for long fits.

## Documentation

- README.md: what the analysis is, the figure, how to run it, short results.
- docs/pipeline.md: per step: what and why, outputs, results, validation.
- docs/differences.md: differences from the authors, limitations, open questions.
- docs/sources.md: inputs' provenance and where each rule lives in the authors' code.

Any change to a step updates the matching docs. Plain, readable prose; no
heavy formatting.

## Workflow

- Explain a change (what, how, why, its source) before or right after making
  it; walk through new code before committing.
- Validate each step against the authors' code where possible, and record the
  result briefly in docs/pipeline.md.
- Never commit unreviewed code. One commit per step or change; ask before
  pushing.
