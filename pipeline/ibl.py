"""The raw IBL data of one session: trials, neurons and their spikes, wheel, whisking and licks.

Read from IBL's public server with IBL's tools (ONE, brainbox), as the
Posani-Wang code does (utils/download_data.py), each file in its last revision
dated on or before a given date (`as_of`). Each session's files
are downloaded to CACHE and deleted once read. The only module that reads
files: the rest of the pipeline gets the data from here.
"""
import shutil
from functools import cache

import numpy as np
import pandas as pd
from brainbox.behavior.wheel import interpolate_position
from brainbox.io.one import SessionLoader
from iblatlas.regions import BrainRegions
from one.api import ONE

# ---- IBL's public server
SERVER = {"base_url": "https://openalyx.internationalbrainlab.org",
          "username": "intbrainlab", "password": "international"}
CACHE = "outputs/ibl_cache"  # downloads, deleted after each session

# ---- Neurons
SPIKE_SORTER = "pykilosort"  # the spike sorting of the Brain Wide Map (Posani-Wang code)
GOOD_LABEL = 1               # IBL's quality label of the units that pass every quality metric

# ---- Wheel
WHEEL_RATE = 100  # Hz: the wheel position is interpolated at this rate (Posani-Wang code)


# ================================================================ server

@cache
def _one():
    """The connection to IBL's server (made once)."""
    return ONE(**SERVER, cache_dir=CACHE, silent=True)


@cache
def _regions():
    """The Allen atlas' brain regions, to name each neuron's area (made once)."""
    return BrainRegions()


# ================================================================ neurons

def _probe_units(eid, probe, as_of):
    """The neurons of one probe and their spikes.

    The same as IBL's SpikeSortingLoader with merge_clusters (Posani-Wang code),
    downloading only the spike files used here. A neuron's area is its channel's
    location in the Allen atlas, mapped to Beryl areas (Posani-Wang code).
    Returns a table with one row per neuron, each spike's time, and the row of
    the neuron that fired it.
    """
    one, collection = _one(), f"alf/{probe}/{SPIKE_SORTER}"
    # The probe's spike sorting, three IBL objects (files), each in its revision as of `as_of`:
    #   spikes:   each spike's time, and the cluster (neuron) that fired it
    #   clusters: each cluster's metrics (IBL cluster id, quality label, firing rate),
    #             the channel it was detected on, and its IBL id
    #   channels: each channel's location, as an Allen atlas region id
    spikes = one.load_object(eid, "spikes", collection=collection,
                             attribute=["times", "clusters"], revision=as_of)
    clusters = one.load_object(eid, "clusters", collection=collection,
                               attribute=["metrics", "channels", "uuids"], revision=as_of)
    channels = one.load_object(eid, "channels", collection=collection,
                               attribute=["brainLocationIds_ccf_2017"], revision=as_of)
    # A cluster's area is the region of its channel.
    area_id = channels["brainLocationIds_ccf_2017"][clusters["channels"]]
    # One row per cluster, in IBL's order: spikes["clusters"] is the row of each spike's cluster.
    units = pd.DataFrame({
        "cluster_id": clusters["metrics"].cluster_id.to_numpy(),
        "uuid": np.asarray(clusters["uuids"]),
        "label": clusters["metrics"].label.to_numpy(),
        "firing_rate": clusters["metrics"].firing_rate.to_numpy(),
        "beryl_acronym": _regions().acronym2acronym(_regions().id2acronym(area_id), mapping="Beryl"),
    })
    return units, spikes["times"], spikes["clusters"]


def _session_units(eid, as_of, good_units, pids=None):
    """Merge the neurons and spikes of a session's probes into one table and one list.

    All its probes, or only those in `pids`. With good_units, only the units
    IBL labels good, and their spikes, are kept.
    Returns `units` (one row per neuron, with its probe id `pid`), `times`
    (seconds, sorted) and `neuron` (row position in `units` of the neuron that
    fired each spike).
    """
    session_pids, probes = _one().eid2pid(eid)  # the session's probes: ids and names (probe00, ...)
    units, times, neuron = [], [], []
    for pid, probe in zip(session_pids, probes):
        if pids is not None and str(pid) not in pids:
            continue
        probe_units, spike_times, spike_cluster = _probe_units(eid, probe, as_of)
        keep = (probe_units.label == GOOD_LABEL).to_numpy() if good_units else np.ones(len(probe_units), bool)
        # Row of each kept cluster in the session's table (-1: not kept). The rows
        # continue after the previous probes' neurons.
        row = np.full(len(probe_units), -1)
        row[keep] = np.arange(keep.sum()) + sum(len(u) for u in units)
        # The spikes of the kept clusters, each labelled with its neuron's row.
        kept = row[spike_cluster] >= 0
        units.append(probe_units[keep].assign(pid=str(pid)))
        times.append(spike_times[kept])
        neuron.append(row[spike_cluster][kept])
    times, neuron = np.concatenate(times), np.concatenate(neuron)
    order = np.argsort(times, kind="stable")  # probes were merged: sort in time again
    return pd.concat(units, ignore_index=True), times[order], neuron[order]


# ================================================================ session

def load_session(eid, as_of, good_units, pids=None):
    """Everything the pipeline uses from one session, each file as of the date `as_of`.

    The neurons come from all the session's probes, or only those in `pids`.

    Returns a dict with:
      trials        the session's trials table, one row per trial (trial_id: its index)
      units         its neurons, one row each (all units, or good units only)
      spike_times   every spike of those neurons (s, sorted), and
      spike_neuron  the row in `units` of the neuron that fired it
      wheel         {"t", "position"}: the wheel's position, sampled at WHEEL_RATE
      cameras       per camera (left, right), {"t", "whisker"}: its whisker motion energy
      lick_times    IBL's lick times (s), as in the Posani-Wang code
    Then deletes the session's downloaded files.
    """
    one = _one()
    try:
        # Trials and whisker motion energy: IBL's SessionLoader, as in the Posani-Wang code.
        session = SessionLoader(one=one, eid=eid, revision=as_of)
        session.load_trials()
        session.load_motion_energy(views=["left", "right"])
        units, spike_times, spike_neuron = _session_units(eid, as_of, good_units, pids)
        # The wheel's position is recorded at irregular times (at each tick of its
        # encoder): IBL's interpolate_position resamples it at WHEEL_RATE.
        wheel = one.load_object(eid, "wheel", collection="alf", attribute=["timestamps", "position"],
                                revision=as_of)
        position, t = interpolate_position(wheel["timestamps"], wheel["position"], freq=WHEEL_RATE)
        return {
            "trials": session.trials.rename_axis("trial_id").reset_index(),  # row number -> trial_id
            "units": units,
            "spike_times": spike_times,
            "spike_neuron": spike_neuron,
            "wheel": {"t": t, "position": position},
            "cameras": {camera: {"t": me.times.to_numpy(), "whisker": me.whiskerMotionEnergy.to_numpy(float)}
                        for camera, me in session.motion_energy.items()},
            "lick_times": one.load_dataset(eid, "licks.times.npy", collection="alf", revision=as_of),
        }
    finally:  # also when the session can't be read
        shutil.rmtree(one.eid2path(eid), ignore_errors=True)
