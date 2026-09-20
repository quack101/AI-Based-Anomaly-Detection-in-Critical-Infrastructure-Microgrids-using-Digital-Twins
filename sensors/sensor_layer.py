"""
The Layer-1 to Layer-2 bridge (RES.md D2, D6.0): z = h(x_true) + e.

build_measurement_vector() is the ONLY place z gets constructed, and it
does so in exactly ONE pass over the ordered, enabled registry entries
(registry.enabled_entries()) -- per entry: read h_k(x_true) via
measurement_functions.measure(), then add per-sensor Gaussian noise
using THAT ENTRY'S OWN sigma (never a global constant). The returned
z[i] corresponds to entries[i] by construction, not by a second,
separately-ordered pass -- see tests/test_sensor_layer.py's ordering
test for what this guards against.

Requires a compiled, solved OpenDSS circuit already active in the
process (core.ieee123_model.compile_and_solve() / extract_all()) --
this module does not solve anything itself.
"""

import random

from sensors.measurement_functions import measure


def build_measurement_vector(entries, add_noise=True, rng=None):
    """entries: the output of registry.enabled_entries(registry) --
    already filtered and ordered. Returns (z, entry_ids), with
    z[i] == h(entries[i]) [+ noise], entry_ids[i] == entries[i]['id'],
    built in the SAME loop -- not two separate ones."""

    if add_noise and rng is None:
        rng = random.Random()

    z = []
    entry_ids = []

    for entry in entries:
        value = measure(entry)

        if add_noise:
            value += rng.gauss(0.0, entry["sigma"])

        z.append(value)
        entry_ids.append(entry["id"])

    return z, entry_ids


def build_measurement_dict(entries, add_noise=True, rng=None):
    """Same as build_measurement_vector, keyed by entry id -- convenient
    for telemetry payloads / debugging; still built from the identical
    single pass (calls build_measurement_vector, does not re-iterate
    the registry)."""

    z, entry_ids = build_measurement_vector(entries, add_noise=add_noise, rng=rng)
    return dict(zip(entry_ids, z))
