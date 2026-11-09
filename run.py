"""The main pipeline: from a dataset's raw IBL data to every neuron's selectivity profile.

Four stages, each saving a checkpoint in outputs/<dataset>/ that the next one reads:
  sessions  selected trials and neurons, binned spikes and behaviour  (pipeline.sessions)
  inputs    the model's targets y and inputs X                         (pipeline.inputs)
  model     the RRR fits: one per cross-validation split, one on all trials  (pipeline.rrr)
  profiles  one row per neuron (area, R², selection) and its coefficients    (pipeline.profiles)
A stage whose output already exists is skipped; the sessions and the model's
fits are checked one by one, so an interrupted run resumes where it stopped.
Delete a stage's folder to redo it. Each stage that runs also draws its check
figure (analyses/checks.py), and the sessions and inputs stages write a summary
table, all in outputs/<dataset>/results/pipeline/.

What goes into the pipeline is a dataset, defined in DATASETS: its data
parameters (which sessions or probes, IBL's data as of which date, which units,
and the sessions excluded by hand). The method's constants are in each pipeline module,
with their sources.

Run from the repo root: python run.py <dataset>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyses import checks
from pipeline import ibl, inputs, profiles, rrr, sessions

# ---- Outputs: the stages' summary tables and check figures, within outputs/<dataset>/
RESULTS = "results/pipeline"

# ---- Datasets: what goes into the pipeline, by name
# as_of: for each IBL file, its last revision dated on or before that date.
DATASETS = {
    "posani_wang": {
        "description": "Posani-Wang's data: IBL's data before the 2024 re-run, all units",
        "sessions": "data/eid_list.npy",  # the 178 sessions Posani-Wang trained on
        # Posani-Wang's trained model has neurons from a probe revised on 2022-08-04 and none
        # from the revisions of 2024-03-22. The last day before 2024-03-22 reproduces their
        # neurons (docs/sources.md).
        "as_of": "2024-03-21",
        "good_units": False,  # Posani-Wang code: every unit
        "excluded_sessions": "data/posani_wang_excluded_sessions.csv",  # by hand, each with its reason
    },
    "bwm": {
        "description": "IBL's current data (2024 neuron lists), good units only",
        # The Brain Wide Map's 459 sessions, and in each only its probes that pass IBL's
        # quality control (699 probes; docs/beyond_cortex.md).
        "probes": "data/bwm_2023_12_release.csv",
        "as_of": "2026-11-01",  # when this dataset was defined: later revisions don't change it
        "good_units": True,     # only the units that pass all of IBL's quality metrics
    },
}


def build_sessions(dataset, out):
    """The sessions checkpoint: for each session, its selected trials, neurons and behaviour, binned."""
    # 1. The dataset's sessions (and, if given, which probes of each), and the ones excluded by hand.
    if "probes" in dataset:
        probes = pd.read_csv(dataset["probes"])
        eids = probes.eid.unique()
        pids = probes.groupby("eid").pid.apply(list).to_dict()
    else:  # every probe of each session
        eids = np.load(dataset["sessions"], allow_pickle=True).astype(str)
        pids = {}
    excluded = (set(pd.read_csv(dataset["excluded_sessions"]).eid)
                if "excluded_sessions" in dataset else set())
    print(f"Sessions: {len(eids)}, excluded by hand: {len(excluded)}")

    # 2. For each session: read its data, prepare trials, neurons and behaviour,
    #    and save the result. Sessions saved in an earlier run are not read again.
    Path(f"{out}/sessions").mkdir(exist_ok=True)
    summary = []
    for i, eid in enumerate(eids):
        first = i == 0  # print the per-rule counts for the first session only
        path = f"{out}/sessions/{eid}.npz"
        if Path(path).exists():
            spikes = np.load(path)["spikes"]
            summary.append({"eid": eid, "status": "saved", "trials": spikes.shape[0],
                            "neurons": spikes.shape[2]})
            continue
        if eid in excluded:
            summary.append({"eid": eid, "status": "excluded"})
            continue

        # ---- Read this session's data (skipped if IBL can't serve it, as in the Posani-Wang code)
        try:
            raw = ibl.load_session(eid, dataset["as_of"], dataset["good_units"], pids.get(eid))
        except Exception as error:
            print(f"{i + 1:3d}/{len(eids)} {eid}: skipped, {type(error).__name__}: {error}")
            summary.append({"eid": eid, "status": "no data"})
            continue

        # ---- Trials
        trials = sessions.prepare_trials(raw["trials"], verbose=first)
        if trials is None:
            print(f"{i + 1:3d}/{len(eids)} {eid}: skipped, fewer than {sessions.MIN_TRIALS} trials")
            summary.append({"eid": eid, "status": "too few trials"})
            continue
        stim_times = trials.stimOn_times.to_numpy()

        # ---- Neurons
        units, spikes = sessions.prepare_neurons(raw["units"], raw["spike_times"], raw["spike_neuron"],
                                                 stim_times, verbose=first)
        if len(units) < sessions.MIN_NEURONS_PER_SESSION:
            print(f"{i + 1:3d}/{len(eids)} {eid}: skipped, only {len(units)} neurons")
            summary.append({"eid": eid, "status": "too few neurons"})
            continue

        # ---- Behaviour
        behaviour = sessions.prepare_behaviour(raw["wheel"], raw["cameras"], raw["lick_times"], stim_times)

        # ---- Save (raw values only: turning them into model inputs is the next stage)
        np.savez_compressed(
            path,
            spikes=spikes,                                  # (trials, 100 bins, neurons) counts
            **behaviour,                                    # wheel, whisker, lick: (trials, 120 bins)
            **{col: trials[col].to_numpy() for col in sessions.TRIAL_COLUMNS},
            area=units.beryl_acronym.to_numpy(str),         # one entry per neuron
            pid=units.pid.to_numpy(str),
            cluster_id=units.cluster_id.to_numpy(),
            uuid=units.uuid.to_numpy(str),                  # IBL's id of each neuron
            # Time axis, so later stages read it from here.
            dt=sessions.DT, neural_start=sessions.NEURAL_START, behav_start=sessions.BEHAV_START,
        )
        print(f"{i + 1:3d}/{len(eids)} {eid}: {len(trials)} trials, {len(units)} neurons")
        summary.append({"eid": eid, "status": "saved", "trials": len(trials),
                        "neurons": len(units)})

    # 3. Summary of the whole stage, also saved as a table.
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{out}/{RESULTS}/sessions.csv", index=False)
    saved = summary[summary.status == "saved"]
    print(f"\nSaved {len(saved)} of {len(eids)} sessions: {saved.trials.sum():.0f} trials, "
          f"{saved.neurons.sum():.0f} neurons. "
          f"Skipped: {summary.status.value_counts().drop('saved', errors='ignore').to_dict()}")


def build_inputs(out):
    """The inputs checkpoint: for each session, the model's targets y and inputs X."""
    # 1. The sessions saved by the sessions stage.
    paths = sorted(Path(f"{out}/sessions").glob("*.npz"))
    print(f"Sessions: {len(paths)}")

    # 2. For each session: build y and X and save them.
    Path(f"{out}/inputs").mkdir()
    summary = []
    for i, path in enumerate(paths):
        # ---- Read the sessions stage's file
        session = np.load(path)
        spikes = session["spikes"]
        dt = float(session["dt"])  # seconds per time bin, set in the sessions stage

        # ---- Targets
        y, y_mean, y_std = inputs.prepare_targets(spikes)

        # ---- Inputs
        x, shifts = inputs.prepare_inputs(session, spikes)

        # ---- Save (not compressed: z-scored decimals barely compress)
        np.savez(
            f"{out}/inputs/{path.stem}.npz",
            X=x.astype(np.float32),                 # (trials, 100 bins, 8 variables)
            y=y.astype(np.float32),                 # (trials, 100 bins, neurons)
            y_mean=y_mean.astype(np.float32),       # (100 bins, neurons), to undo the z-score
            y_std=y_std.astype(np.float32),
            variables=np.array(inputs.VARIABLES),   # names of X's 8 columns
            **{f"{name}_cross_corr": shifts[name][1] for name in inputs.MOVEMENTS},  # one value per shift
            area=session["area"], pid=session["pid"], cluster_id=session["cluster_id"],
            uuid=session["uuid"],
            dt=session["dt"], neural_start=session["neural_start"],  # time axis, passed on
        )
        shift_ms = {f"{name}_shift_ms": round(shifts[name][0] * dt * 1000) for name in inputs.MOVEMENTS}
        # A failed search (best shift at the edge of the range) also gives 0 ms.
        failed = {f"{name}_search_failed":
                  np.argmax(shifts[name][1]) in (0, len(shifts[name][1]) - 1)
                  for name in inputs.MOVEMENTS}
        print(f"{i + 1:3d}/{len(paths)} {path.stem}: X {x.shape}, y {y.shape}, shifts (ms) "
              + ", ".join(f"{name} {shift_ms[f'{name}_shift_ms']:+d}" for name in inputs.MOVEMENTS))
        summary.append({"eid": path.stem, "trials": y.shape[0], "neurons": y.shape[2],
                        **shift_ms, **failed})

    # 3. Summary of the whole stage, also saved as a table.
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{out}/{RESULTS}/shifts.csv", index=False)
    print(f"\nPreprocessed {len(summary)} sessions. Failed shift searches (no shift used): "
          + ", ".join(f"{name} {summary[f'{name}_search_failed'].sum()}" for name in inputs.MOVEMENTS))


def _open_inputs(out):
    """Every session's inputs (the inputs checkpoint), by session id. Used by the model and the profiles.

    The files are opened, not read: each array is read from disk when it is
    used, so only one session is in memory at a time. Each fit converts the
    trials it uses to float64, as in the Posani-Wang code.
    """
    return {path.stem: np.load(path) for path in sorted(Path(f"{out}/inputs").glob("*.npz"))}


def fit_model(out):
    """The model checkpoint: the split fits (test R² per neuron) and the fit on all trials.

    Each fit is saved as soon as it is done and skipped if its file already
    exists, so an interrupted run resumes where it stopped.
    """
    # 1. Every session's inputs.
    data = _open_inputs(out)
    print(f"Sessions: {len(data)}, neurons: {sum(d['y'].shape[2] for d in data.values())}")
    Path(f"{out}/model").mkdir(exist_ok=True)

    # 2. The split fits: fit on 70% of the trials, measure each neuron's R² on the other 30%.
    for split in range(rrr.N_SPLITS):
        path = f"{out}/model/split{split}.npz"
        if not Path(path).exists():  # otherwise it was done in an earlier run
            r2, losses = rrr.cross_validated_r2(data, split)
            np.savez(path, **r2, losses=losses)  # one R² per neuron, for each session

    # 3. The final fit, on all trials: its coefficients go to the profiles.
    path = f"{out}/model/all_trials.npz"
    if not Path(path).exists():
        model, losses = rrr.fit_all_trials(data)
        np.savez(path, V=model.V.detach().numpy(), losses=losses,
                 **{f"U_{eid}": model.U[eid].detach().numpy() for eid in data},
                 **{f"b_{eid}": model.b[eid].detach().numpy() for eid in data})

    # 4. Summary.
    splits = [np.load(f"{out}/model/split{split}.npz") for split in range(rrr.N_SPLITS)]
    r2 = np.concatenate([np.mean([s[eid] for s in splits], axis=0) for eid in data])
    print(f"Done. Per-neuron test R²: median {np.median(r2):.3f}, "
          f"above 0 for {(r2 > 0).mean():.0%} of {len(r2)} neurons.")


def build_profiles(out):
    """The profiles checkpoint: one row per neuron, and its coefficients beta in the same order."""
    # 1. The model's results: the per-neuron test R² of each split, and the final model.
    data = _open_inputs(out)
    splits = [np.load(f"{out}/model/split{split}.npz") for split in range(rrr.N_SPLITS)]
    final = np.load(f"{out}/model/all_trials.npz")

    # 2. For each neuron: R² of the RRR and of the null model; and the coefficients.
    table = profiles.neuron_table(data, splits)
    beta = np.concatenate([final[f"U_{eid}"] @ final["V"] for eid in data])  # (neurons, variables, time bins)

    # 3. Save: one row per neuron, and its coefficients in the same order.
    Path(f"{out}/profiles").mkdir()
    table.to_csv(f"{out}/profiles/table.csv", index=False)
    np.save(f"{out}/profiles/beta.npy", beta.astype(np.float32))

    # 4. Summary.
    print(f"Neurons: {len(table)}. Mean R²: RRR {table.r2_rrr.mean():.3f}, "
          f"null {table.r2_null.mean():.3f}. Selected (delta R² > {profiles.MIN_DELTA_R2}): "
          f"{table.selected.sum()}")


if __name__ == "__main__":
    name = sys.argv[1]
    dataset = DATASETS[name]
    print(f"Dataset {name}: {dataset['description']}")
    out = f"outputs/{name}"
    Path(f"{out}/{RESULTS}").mkdir(parents=True, exist_ok=True)

    # Each stage, unless its checkpoint exists (the sessions and the model check
    # each file themselves), then its check figure.
    stages = [("sessions", lambda: build_sessions(dataset, out), checks.check_sessions),
              ("inputs", lambda: build_inputs(out), checks.check_inputs),
              ("model", lambda: fit_model(out), checks.check_model),
              ("profiles", lambda: build_profiles(out), checks.check_profiles)]
    for stage, run, check in stages:
        if stage not in ("sessions", "model") and Path(f"{out}/{stage}").exists():
            print(f"\n== {stage}: already in {out}/{stage}/, skipped")
            continue
        print(f"\n== {stage}")
        run()
        check(out, f"{out}/{RESULTS}")
