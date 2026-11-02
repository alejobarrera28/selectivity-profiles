"""The sessions stage: turn each session's raw IBL data into model-ready arrays.

For each session, three functions turn the raw data (from pipeline.ibl) into
what the model needs:
  prepare_trials    -> the selected trials (and their task variables)
  prepare_neurons   -> spike counts of the selected neurons in 10 ms bins
  prepare_behaviour -> wheel, whisker and lick signals in 10 ms bins
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

# ---- Time axis (Posani-Wang Methods): -0.2 to 0.8 s around stimulus onset, 10 ms bins
DT = 0.01
NEURAL_START, NEURAL_BINS = -0.2, 100
# Behaviour gets 10 extra bins on each side (-0.3 to 0.9 s, 120 bins), so that
# the inputs can shift each movement trace in time by up to +-100 ms to line it
# up with the neural activity (Posani-Wang Methods; the Posani-Wang code does the same).
BEHAV_START, BEHAV_BINS = -0.3, 120

# ---- Trials
MAX_REACTION_TIME = 0.8  # seconds (Posani-Wang Methods, "Criteria for trial inclusion")
MIN_TRIALS = 100         # Posani-Wang code: sessions with fewer selected trials are skipped
# Trial table columns kept for the inputs, which turn them into the model's task
# variables: block (probabilityLeft), side and contrast (contrastLeft/Right),
# choice, and outcome (feedbackType).
TRIAL_COLUMNS = ["trial_id", "probabilityLeft", "contrastLeft", "contrastRight",
                 "choice", "feedbackType"]

# ---- Neurons
MIN_FR, MAX_FR = 0.5, 50.0      # Hz (Posani-Wang Methods, "Criteria for neuron inclusion")
MAX_FRAC_SILENT_TRIALS = 0.5    # Posani-Wang code: must spike in at least half of the trials
MIN_NEURONS_PER_SESSION = 5     # Posani-Wang code: sessions with fewer neurons are skipped

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
        "made a choice": trials.choice != 0,                                                  # Posani-Wang code
        "moved after the stimulus": reaction_time >= 0,                                       # Posani-Wang code
        f"started moving within {MAX_REACTION_TIME} s": reaction_time < MAX_REACTION_TIME,    # Posani-Wang Methods
        f"finished response within {MAX_REACTION_TIME} s": response_time < MAX_REACTION_TIME, # Posani-Wang code
        # The 50/50 block only occurs in the first ~90 trials of each session,
        # so it is confounded with time in the session.
        "not in the 50/50 block": trials.probabilityLeft != 0.5,                              # Posani-Wang Methods
    }
    if verbose:
        _log_rules(rules, "trials")
    keep = np.logical_and.reduce(list(rules.values()))
    if keep.sum() < MIN_TRIALS:
        return None
    return trials[keep]


# ================================================================ neurons

def prepare_neurons(units, spike_times, spike_neuron, stim_times, verbose=False):
    """Bin the spikes of one session and select the neurons.

    `units` has one row per neuron; spike_neuron[i] is the row of the neuron
    that fired the spike at spike_times[i]. Spikes are counted in NEURAL_BINS
    bins of DT around each stimulus; a neuron must pass all the rules below.
    Returns the kept rows of `units` and their spike counts (n_trials, n_bins, n_neurons).
    """
    spikes = _count_in_bins(spike_times, spike_neuron, len(units), stim_times,
                            NEURAL_START, NEURAL_BINS)

    per_trial = spikes.sum(axis=1)                          # spikes per trial
    task_rate = per_trial.mean(axis=0) / (NEURAL_BINS * DT)  # Hz
    frac_silent = (per_trial == 0).mean(axis=0)             # fraction of trials with no spikes
    rules = {
        # Units without an area label are what the Posani-Wang atlas mapping calls
        # "root"; "y" is a fiber tract.
        "in a real brain area":                                      # Posani-Wang code
            (units.beryl_acronym.notna() & (units.beryl_acronym != "y")).to_numpy(),
        f"session rate {MIN_FR}-{MAX_FR} Hz":                        # Posani-Wang Methods
            units.firing_rate.between(MIN_FR, MAX_FR, inclusive="neither").to_numpy(),
        f"task-window rate > {MIN_FR} Hz":                           # Posani-Wang code
            task_rate > MIN_FR,
        f"silent in < {MAX_FRAC_SILENT_TRIALS:.0%} of trials":       # Posani-Wang code
            frac_silent < MAX_FRAC_SILENT_TRIALS,
    }
    if verbose:
        _log_rules(rules, "neurons")
    keep = np.logical_and.reduce(list(rules.values()))
    return units[keep], spikes[:, :, keep]


# ================================================================ behaviour

def _sample_on_bins(t, x, stim_times):
    """Value of a continuous signal x(t) at the start of each behaviour bin:
    (n_trials, BEHAV_BINS). Linear interpolation between samples, as the Posani-Wang code does.
    """
    bin_starts = stim_times[:, None] + BEHAV_START + DT * np.arange(BEHAV_BINS)
    return np.interp(bin_starts, t, x)


def _wheel_velocity(t, position):
    """Wheel velocity (rad/s) from position sampled uniformly at 100 Hz.

    Same as IBL's brainbox.behavior.wheel.velocity_filtered, which the
    Posani-Wang code uses: low-pass the position (8th-order Butterworth, 20 Hz, forwards and
    backwards so there is no delay), then take the derivative.
    """
    fs = 1 / np.median(np.diff(t))
    sos = butter(WHEEL_FILTER_ORDER, WHEEL_CUTOFF / (fs / 2), btype="lowpass", output="sos")
    return np.gradient(sosfiltfilt(sos, position), t)


def _detect_licks(t, tongue):
    """Lick times from the tongue keypoints of one camera.

    Same as IBL's brainbox.behavior.dlc.get_licks, which produced the lick
    times Posani-Wang used: ignore keypoints tracked with likelihood < 0.9, then
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


def prepare_behaviour(wheel, cameras, stim_times):
    """Wheel velocity, whisker motion energy and lick counts of one session.

    `wheel` and `cameras` are as returned by ibl.load_session. Each output is
    an array (n_trials, BEHAV_BINS).
    """
    wheel = _sample_on_bins(wheel["t"], _wheel_velocity(wheel["t"], wheel["position"]), stim_times)

    whisker, lick_times = [], []
    for camera in cameras.values():
        t, features = camera["t"], camera["features"]
        whisker.append(_sample_on_bins(t, features.whiskerMotionEnergy.to_numpy(float), stim_times))
        lick_times.append(_detect_licks(t, features))
    # One whisker signal: the larger of the two cameras at each time (Posani-Wang code).
    whisker = np.maximum(*whisker)
    # Licks seen by either camera, counted per bin (as IBL combines the two cameras).
    lick_times = np.unique(np.concatenate(lick_times))
    licks = _count_in_bins(lick_times, np.zeros(len(lick_times), int), 1,
                          stim_times, BEHAV_START, BEHAV_BINS)[:, :, 0]
    return {"wheel": wheel, "whisker": whisker, "lick": licks}
