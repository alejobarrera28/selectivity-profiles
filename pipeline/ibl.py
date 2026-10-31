"""The raw IBL data of one session: trials, neurons and their spikes, wheel and cameras.

Read from the Brain Wide Map datasets the IBL agent downloaded: tables
(.parquet) with pandas, spikes and behaviour traces with the agent's two
decoders, since the agent stored them in its own compressed format. The only
module that reads files: the rest of the pipeline gets the data from here.
"""
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from ibl_ai_agent.datasets.bwm_behavior import load_behavior_session_shard
from ibl_ai_agent.datasets.bwm_ephys import load_spike_shard

# ---- Data location: the datasets the IBL agent downloaded
DATA = "../ibl-ai-agent/reports/datasets"
EPHYS = f"{DATA}/bwm_ephys/1.2.1"          # trials, units and spikes
BEHAVIOUR = f"{DATA}/bwm_behavior/2.0.0"   # wheel and camera traces


# ================================================================ tables

@cache
def _tables():
    """The trials and units tables of every session (read once)."""
    return (pd.read_parquet(f"{EPHYS}/metadata/trials.parquet"),
            pd.read_parquet(f"{EPHYS}/metadata/units.parquet"))


def available_sessions():
    """The ids (eid) of the sessions in the data."""
    return set(_tables()[0].eid)


# ================================================================ session

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


def load_session(eid):
    """Everything the pipeline uses from one session.

    Returns a dict with:
      trials        the session's rows of the trials table
      units         its neurons (good units only), one row each
      spike_times   every spike of those neurons (s, sorted), and
      spike_neuron  the row in `units` of the neuron that fired it
      wheel         {"t", "position"}: the wheel's position over time
      cameras       per camera, {"t", "features"}: its tracked features over time
    """
    all_trials, all_units = _tables()
    units = all_units[all_units.eid == eid].reset_index(drop=True)
    spike_shards = {pid: load_spike_shard(Path(f"{EPHYS}/spikes/{pid}"))
                    for pid in units.pid.unique()}   # one spike file per probe
    spike_times, spike_neuron = _merge_probes(units, spike_shards)
    shard = load_behavior_session_shard(Path(f"{BEHAVIOUR}/sessions/{eid}.zip"))
    return {
        "trials": all_trials[all_trials.eid == eid],
        "units": units,
        "spike_times": spike_times,
        "spike_neuron": spike_neuron,
        "wheel": {"t": shard["wheel.timestamps"],
                  "position": shard["wheel.position"].astype(float)},
        "cameras": {camera: {"t": shard[f"{camera}.timestamps"],
                             "features": pd.DataFrame(
                                 shard[f"{camera}.features"],
                                 columns=shard["meta"]["cameras"][camera]["columns"])}
                    for camera in ("leftCamera", "rightCamera")},
    }
