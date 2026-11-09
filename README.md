# Selectivity profiles of neurons across the mouse brain

How do single neurons across the whole mouse brain respond to the task,
behaviour and context variables of the IBL decision-making task, and how do
these selectivity profiles differ between brain regions? Work in progress.

Each neuron's activity is modelled with a reduced-rank regression (RRR)
encoding model, fitted to the IBL Brain Wide Map; its coefficients give the
neuron's selectivity profile. The model follows Posani, Wang et al. (Nature
2026), reimplemented from scratch.

## Code

- `run.py <dataset>`: the pipeline, from IBL data to every neuron's profile,
  in four stages (sessions, inputs, model, profiles), one module each in
  `pipeline/`; checkpoints in `outputs/<dataset>/`.
- `analyses/`: scripts that read the checkpoints: check figures for each
  stage, and the analyses (so far, mean selectivity per area, and the
  comparison with Posani-Wang's trained model).
- `docs/`: notes on each stage, its choices and their sources.

## Running it

Requirements: Python 3.12 and an internet connection: the data is read from
IBL's public server, one session at a time, and deleted once read.

Create the environment once, from this folder:

    conda create -n selectivity-profiles python=3.12
    conda activate selectivity-profiles
    pip install -r requirements.txt

Then, with the environment active:

    caffeinate -i python run.py bwm            # the pipeline (macOS: caffeinate keeps the Mac awake)
    python -m analyses.area_selectivity bwm    # mean selectivity per area

`bwm` is a dataset, defined in `run.py`: which sessions, IBL's data as of
which date, and which units. `bwm` is the IBL Brain Wide Map with IBL's
current data, good units only; `shuqi` is Posani-Wang's data. Reading the data
takes a few hours and fitting the model about half an hour per dataset. A stage
whose output exists is skipped, and the sessions and the model's fits are
saved one by one, so running it again after an interruption resumes where it
left off. Figures and tables, kept in the repo, go to
`outputs/<dataset>/results/`: the pipeline's summaries and a check figure for
each stage to `pipeline/`, and each analysis's to a folder of its own.

`requirements.txt` pins the versions the results were produced with.
