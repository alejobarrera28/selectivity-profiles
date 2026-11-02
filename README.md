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
  stage, and the analyses (so far, mean selectivity per area).
- `docs/`: notes on each stage, its choices and their sources.

## Running it

Requirements: Python 3.12, and the IBL Brain Wide Map datasets as downloaded
by the IBL AI agent, in a sibling folder:

    git clone https://github.com/int-brain-lab/ibl-ai-agent ../ibl-ai-agent
    # then download bwm_ephys and bwm_behavior (~9 GB) into
    # ../ibl-ai-agent/reports/datasets/, following the agent's README

Create the environment once, from this folder:

    conda create -n selectivity-profiles python=3.12
    conda activate selectivity-profiles
    pip install -r requirements.txt

Then, with the environment active:

    caffeinate -i python run.py agent           # the pipeline: about 1 hour (macOS: keeps the Mac awake)
    python -m analyses.checks agent             # check figures for each stage
    python -m analyses.area_selectivity agent   # mean selectivity per area

`agent` is the dataset, defined in `run.py`. The code expects `bwm_ephys`
1.2.1 and `bwm_behavior` 2.0.0; with a newer download, update the `EPHYS` and
`BEHAVIOUR` paths at the top of `pipeline/ibl.py`. A stage whose output exists
is skipped, and the model saves each fit as soon as it is done, so running it
again after an interruption resumes where it left off. Figures and tables go
to `outputs/<dataset>/results/`, which is kept in the repo.

`requirements.txt` pins the versions the results were produced with.
`ibl-ai-agent` is only used to decode the agent's compressed data files;
`brainwidemap` is a dependency of the agent, only available on GitHub.
