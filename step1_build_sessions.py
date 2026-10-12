"""Step 1: turn each session's raw IBL data into model-ready arrays.

All data is read in the main block, at the bottom. Then, for each session,
three functions turn the raw data into what the model needs:
  prepare_trials    -> the selected trials (and their task variables)
  prepare_neurons   -> spike counts of the selected neurons in 10 ms bins
  prepare_behaviour -> wheel, whisker and lick signals in 10 ms bins

Tables (.parquet) are read with pandas. Spikes and behaviour traces were stored
by the IBL agent in its own compressed format, so we read them with the agent's
two decoders (load_spike_shard, load_behavior_session_shard).

Output: one file per session in outputs/sessions/, plus a figure and a
per-session table in outputs/results/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ibl_ai_agent.datasets.bwm_behavior import load_behavior_session_shard
from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard
from scipy.signal import butter, sosfiltfilt

# ---- Data location: the datasets the IBL agent downloaded
DATA = "../ibl-ai-agent/reports/datasets"
EPHYS = f"{DATA}/bwm_ephys/1.2.1"          # trials, units and spikes
BEHAVIOUR = f"{DATA}/bwm_behavior/2.0.0"   # wheel and camera traces

# ---- Outputs
SESSIONS_DIR = "outputs/sessions"   # one .npz per session (used by step 2)
RESULTS_DIR = "outputs/results"     # figure and per-session table (kept in the repo)

# ---- Sessions: the ones the authors trained on
AUTHORS_EIDS = np.load("data/eid_list.npy", allow_pickle=True).astype(str)

# ---- Time axis (paper Methods): -0.2 to 0.8 s around stimulus onset, 10 ms bins
DT = 0.01
NEURAL_START, NEURAL_BINS = -0.2, 100
# Behaviour gets 10 extra bins on each side (-0.3 to 0.9 s, 120 bins), so that
# step 2 can shift each movement trace in time by up to +-100 ms to line it up
# with the neural activity (paper Methods; the authors' code does the same).
BEHAV_START, BEHAV_BINS = -0.3, 120

# ---- Trials
MAX_REACTION_TIME = 0.8  # seconds (paper Methods, "Criteria for trial inclusion")
MIN_TRIALS = 100         # authors' code: sessions with fewer selected trials are skipped
# Trial table columns saved for step 2, which turns them into the model's task
# variables: block (probabilityLeft), side and contrast (contrastLeft/Right),
# choice, and outcome (feedbackType).
TRIAL_COLUMNS = ["trial_id", "probabilityLeft", "contrastLeft", "contrastRight",
                 "choice", "feedbackType"]

# ---- Neurons
MIN_FR, MAX_FR = 0.5, 50.0      # Hz (paper Methods, "Criteria for neuron inclusion")
MAX_FRAC_SILENT_TRIALS = 0.5    # authors' code: must spike in at least half of the trials
MIN_NEURONS = 5                 # authors' code: sessions with fewer neurons are skipped
# The cortical areas the paper analyses. Here only used to report how many
# cortical neurons each session contributes.
CORTICAL_AREAS = pd.read_csv("data/area_list.csv", header=None)[0].tolist()

# ---- Behaviour
WHEEL_FILTER_ORDER, WHEEL_CUTOFF = 8, 20  # IBL's wheel low-pass filter (order, Hz)
LICK_LIKELIHOOD = 0.9                     # IBL's lick detector ignores keypoints below this likelihood
LICK_JUMP_FRACTION = 1 / 4                # IBL's lick = tongue jump > this fraction of the jumps' std


# ================================================================ helpers

def _log_rules(rules, what):
    """Print how many items each rule rejects (an item can fail several rules)."""
    keep = np.logical_and.reduce(list(rules.values()))
    print(f"  {len(keep)} {what} -> {keep.sum()} kept")
    for name, passed in rules.items():
        print(f"    {(~np.asarray(passed)).sum():5d} fail: {name}")


def _count_in_bins(event_times, labels, n_labels, stim_times, start, n_bins):
    """Count events in n_bins bins of DT seconds, starting `start` s after each stimulus.

    event_times must be sorted; labels[i] says which neuron (or 0 for licks)
    event i belongs to. Returns an array (n_trials, n_bins, n_labels).
    """
    counts = np.zeros((len(stim_times), n_bins, n_labels), dtype=np.uint16)
    for k, t0 in enumerate(stim_times):
        first = t0 + start
        lo, hi = np.searchsorted(event_times, [first, first + n_bins * DT])
        # np.minimum: floating-point rounding can put an event at the very end
        # of the window into a bin that doesn't exist (index n_bins).
        bins = np.minimum(((event_times[lo:hi] - first) / DT).astype(int), n_bins - 1)
        np.add.at(counts[k], (bins, labels[lo:hi]), 1)
    return counts


# ================================================================ trials

def prepare_trials(trials, verbose=False):
    """Select the trials of one session that the model will be fitted on.

    `trials` is the session's rows of the trials table; a trial must pass all
    the rules below. Trials with a missing time fail the timing rules
    automatically, since comparisons with NaN are False.
    Returns None if fewer than MIN_TRIALS trials are left.
    """
    reaction_time = trials.firstMovement_times - trials.stimOn_times
    response_time = trials.response_times - trials.stimOn_times
    rules = {
        "made a choice": trials.choice != 0,                                                  # authors' code
        "moved after the stimulus": reaction_time >= 0,                                       # authors' code
        f"started moving within {MAX_REACTION_TIME} s": reaction_time < MAX_REACTION_TIME,    # paper Methods
        f"finished response within {MAX_REACTION_TIME} s": response_time < MAX_REACTION_TIME, # authors' code
        # The 50/50 block only occurs in the first ~90 trials of each session,
        # so it is confounded with time in the session.
        "not in the 50/50 block": trials.probabilityLeft != 0.5,                              # paper Methods
    }
    if verbose:
        _log_rules(rules, "trials")
    keep = np.logical_and.reduce(list(rules.values()))
    if keep.sum() < MIN_TRIALS:
        return None
    return trials[keep]


# ================================================================ neurons

def _merge_probes(units, spike_shards):
    """Merge the spikes of all probes of a session into one list.

    `spike_shards` maps each probe id (pid) to its decoded spike file.
    Returns `times` (seconds, sorted) and `neuron` (row position in `units` of
    the neuron that fired each spike).
    """
    times, neuron = [], []
    for pid, probe_units in units.groupby("pid", sort=False):
        shard = spike_shards[pid]
        # In a shard, spike_clusters[i] is a position in shard["cluster_ids"].
        cluster_of_spike = shard["cluster_ids"][shard["spike_clusters"]]
        row_of_cluster = pd.Series(probe_units.index, index=probe_units.cluster_id)
        times.append(shard["spike_times_seconds"])
        neuron.append(row_of_cluster.loc[cluster_of_spike].to_numpy())
    times, neuron = np.concatenate(times), np.concatenate(neuron)
    order = np.argsort(times, kind="stable")  # probes were merged: sort in time again
    return times[order], neuron[order]


def prepare_neurons(units, spike_shards, stim_times, verbose=False):
    """Bin the spikes of one session and select the neurons.

    `units` is the session's rows of the units table, `spike_shards` its
    decoded spike files (one per probe). Spikes are counted in NEURAL_BINS
    bins of DT around each stimulus; a neuron must pass all the rules below.
    Returns the kept rows of `units` and their spike counts (n_trials, n_bins, n_neurons).
    """
    units = units.reset_index(drop=True)
    times, neuron = _merge_probes(units, spike_shards)
    spikes = _count_in_bins(times, neuron, len(units), stim_times, NEURAL_START, NEURAL_BINS)

    per_trial = spikes.sum(axis=1)                          # spikes per trial
    task_rate = per_trial.mean(axis=0) / (NEURAL_BINS * DT)  # Hz
    frac_silent = (per_trial == 0).mean(axis=0)             # fraction of trials with no spikes
    rules = {
        # Units without an area label are what the authors' atlas mapping calls
        # "root"; "y" is a fiber tract.
        "in a real brain area":                                      # authors' code
            (units.beryl_acronym.notna() & (units.beryl_acronym != "y")).to_numpy(),
        f"session rate {MIN_FR}-{MAX_FR} Hz":                        # paper Methods
            units.firing_rate.between(MIN_FR, MAX_FR, inclusive="neither").to_numpy(),
        f"task-window rate > {MIN_FR} Hz":                           # authors' code
            task_rate > MIN_FR,
        f"silent in < {MAX_FRAC_SILENT_TRIALS:.0%} of trials":       # authors' code
            frac_silent < MAX_FRAC_SILENT_TRIALS,
    }
    if verbose:
        _log_rules(rules, "neurons")
    keep = np.logical_and.reduce(list(rules.values()))
    return units[keep], spikes[:, :, keep]


# ================================================================ behaviour

def _sample_on_bins(t, x, stim_times):
    """Value of a continuous signal x(t) at the start of each behaviour bin:
    (n_trials, BEHAV_BINS). Linear interpolation between samples, as the authors do.
    """
    bin_starts = stim_times[:, None] + BEHAV_START + DT * np.arange(BEHAV_BINS)
    return np.interp(bin_starts, t, x)


def _wheel_velocity(t, position):
    """Wheel velocity (rad/s) from position sampled uniformly at 100 Hz.

    Same as IBL's brainbox.behavior.wheel.velocity_filtered, which the authors
    use: low-pass the position (8th-order Butterworth, 20 Hz, forwards and
    backwards so there is no delay), then take the derivative.
    """
    fs = 1 / np.median(np.diff(t))
    sos = butter(WHEEL_FILTER_ORDER, WHEEL_CUTOFF / (fs / 2), btype="lowpass", output="sos")
    return np.gradient(sosfiltfilt(sos, position), t)


def _detect_licks(t, tongue):
    """Lick times from the tongue keypoints of one camera.

    Same as IBL's brainbox.behavior.dlc.get_licks, which produced the lick
    times the authors used: ignore keypoints tracked with likelihood < 0.9, then
    call a frame a lick event when any tongue coordinate jumps by more than a
    quarter of its typical frame-to-frame change (std of the differences).
    """
    events = []
    for tip in ("tongue_end_l", "tongue_end_r"):
        reliable = tongue[f"{tip}_likelihood"] >= LICK_LIKELIHOOD
        for coord in ("x", "y"):
            x = tongue[f"{tip}_{coord}"].where(reliable).to_numpy()
            jump = np.abs(np.diff(x))
            events.append(np.flatnonzero(jump > np.nanstd(jump) * LICK_JUMP_FRACTION))
    return t[np.unique(np.concatenate(events))]


def prepare_behaviour(shard, stim_times):
    """Wheel velocity, whisker motion energy and lick counts of one session.

    `shard` is the session's decoded behaviour file. Each output is an array
    (n_trials, BEHAV_BINS).
    """
    t = shard["wheel.timestamps"]
    wheel = _sample_on_bins(t, _wheel_velocity(t, shard["wheel.position"].astype(float)), stim_times)

    whisker, lick_times = [], []
    for camera in ("leftCamera", "rightCamera"):
        t = shard[f"{camera}.timestamps"]
        features = pd.DataFrame(shard[f"{camera}.features"],
                                columns=shard["meta"]["cameras"][camera]["columns"])
        whisker.append(_sample_on_bins(t, features.whiskerMotionEnergy.to_numpy(float), stim_times))
        lick_times.append(_detect_licks(t, features))
    # One whisker signal: the larger of the two cameras at each time (authors' code).
    whisker = np.maximum(*whisker)
    # Licks seen by either camera, counted per bin (as IBL combines the two cameras).
    lick_times = np.unique(np.concatenate(lick_times))
    licks = _count_in_bins(lick_times, np.zeros(len(lick_times), int), 1,
                          stim_times, BEHAV_START, BEHAV_BINS)[:, :, 0]
    return {"wheel": wheel, "whisker": whisker, "lick": licks}


# ================================================================ figure

def plot_signals_by_choice(trials, spikes, behaviour, path):
    """Trial-averaged population rate, wheel, whisker and licks, split by choice.

    If binning and alignment are right, the wheel turns in opposite directions
    for the two choices shortly after the stimulus, and licks follow the response.
    """
    t_neural = NEURAL_START + DT * (np.arange(NEURAL_BINS) + 0.5)  # spike counts: bin centres
    t_behav = BEHAV_START + DT * np.arange(BEHAV_BINS)             # behaviour: sampled at bin starts
    cw = trials.choice.to_numpy() == 1
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
    signals = [("population rate (Hz/neuron)", t_neural, spikes.mean(axis=2) / DT)]
    signals += [(name, t_behav, x) for name, x in behaviour.items()]
    for ax, (name, t, x) in zip(axes, signals):
        ax.plot(t, x[cw].mean(0), label="choice +1 (clockwise)")
        ax.plot(t, x[~cw].mean(0), label="choice -1 (counter-clockwise)")
        ax.axvline(0, color="k", lw=0.5)
        ax.set(title=name, xlabel="time from stimulus onset (s)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ================================================================ main

if __name__ == "__main__":
    # 1. Start from everything the IBL agent downloaded: the whole Brain Wide Map
    #    release (all sessions, all trials, all good units).
    all_trials = pd.read_parquet(f"{EPHYS}/metadata/trials.parquet")
    all_units = pd.read_parquet(f"{EPHYS}/metadata/units.parquet")
    print(f"Agent's dataset: {all_trials.eid.nunique()} sessions, "
          f"{len(all_trials)} trials, {len(all_units)} good units")

    # 2. Keep only the sessions the authors trained on, not the whole Brain Wide
    #    Map. A few of their sessions are not in the agent's dataset, so we use
    #    the ones that are.
    eids = [e for e in AUTHORS_EIDS if e in set(all_trials.eid)]
    print(f"Authors' sessions: {len(AUTHORS_EIDS)}, available locally: {len(eids)}")

    # 3. For each session: read its files, prepare trials, neurons and behaviour,
    #    and save the result.
    Path(SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    summary = []
    for i, eid in enumerate(eids):
        first = i == 0  # print the per-rule counts and plot the signals for the first session only

        # ---- Read this session's files
        session_trials = all_trials[all_trials.eid == eid]
        session_units = all_units[all_units.eid == eid]
        spike_shards = {pid: load_spike_shard(Path(f"{EPHYS}/spikes/{pid}"))
                        for pid in session_units.pid.unique()}   # one spike file per probe
        behaviour_shard = load_behavior_session_shard(Path(f"{BEHAVIOUR}/sessions/{eid}.zip"))

        # ---- Trials
        trials = prepare_trials(session_trials, verbose=first)
        if trials is None:
            print(f"{i + 1:3d}/{len(eids)} {eid}: skipped, fewer than {MIN_TRIALS} trials")
            summary.append({"eid": eid, "status": "too few trials"})
            continue
        stim_times = trials.stimOn_times.to_numpy()

        # ---- Neurons
        units, spikes = prepare_neurons(session_units, spike_shards, stim_times, verbose=first)
        if len(units) < MIN_NEURONS:
            print(f"{i + 1:3d}/{len(eids)} {eid}: skipped, only {len(units)} neurons")
            summary.append({"eid": eid, "status": "too few neurons"})
            continue

        # ---- Behaviour
        behaviour = prepare_behaviour(behaviour_shard, stim_times)

        # ---- Save (raw values only: turning them into model inputs is step 2)
        np.savez_compressed(
            f"{SESSIONS_DIR}/{eid}.npz",
            spikes=spikes,                                  # (trials, 100 bins, neurons) counts
            **behaviour,                                    # wheel, whisker, lick: (trials, 120 bins)
            **{col: trials[col].to_numpy() for col in TRIAL_COLUMNS},
            area=units.beryl_acronym.to_numpy(str),         # one entry per neuron
            pid=units.pid.to_numpy(str),
            cluster_id=units.cluster_id.to_numpy(),
            dt=DT, neural_start=NEURAL_START,               # time axis, so later steps read it from here
        )
        n_cortex = units.beryl_acronym.isin(CORTICAL_AREAS).sum()
        print(f"{i + 1:3d}/{len(eids)} {eid}: {len(trials)} trials, "
              f"{len(units)} neurons ({n_cortex} cortical)")
        summary.append({"eid": eid, "status": "saved", "trials": len(trials),
                        "neurons": len(units), "cortical_neurons": n_cortex})

        # ---- Figure (first session only)
        if first:
            plot_signals_by_choice(trials, spikes, behaviour,
                                   f"{RESULTS_DIR}/step1_signals_by_choice.png")

    # 4. Summary of the whole step, also saved as a table.
    summary = pd.DataFrame(summary)
    summary.to_csv(f"{RESULTS_DIR}/step1_sessions.csv", index=False)
    saved = summary[summary.status == "saved"]
    print(f"\nSaved {len(saved)} of {len(eids)} sessions: {saved.trials.sum():.0f} trials, "
          f"{saved.neurons.sum():.0f} neurons ({saved.cortical_neurons.sum():.0f} cortical). "
          f"Skipped: {summary.status.value_counts().drop('saved', errors='ignore').to_dict()}")
