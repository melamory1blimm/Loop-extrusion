#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accelerated cohesin simulator: nine states from registered b/c collision events.

NEW: moving cohesin arms also register a c-blockage at the contacted PASSIVE
arm of the other cohesin (reciprocal contact). This is triggered only by an
actual blocked movement attempt, not by a separate proximity search.
Use --no-passive-c-collisions to reproduce the previous event-classification
behavior with the same random seed. The flag does not change the primary
collision movement rule; it adds the missing c blocker to the passive leg.
Results are tagged pc1/pc0 to avoid accidentally overwriting older data.

Mode 1: one-sided extruder with fixed direction; switching time is irrelevant.
Mode 2: two-sided extruder, both arms moving at half-speed; switching is irrelevant.
Mode 3: one-sided extruder switching direction at Poisson times.
Mode 4: one-sided extruder switching direction at fixed time intervals.

Exports nine mutually exclusive collision states plus A, A_bb, A_b, A_00.
Also saves four individually normalized theory-vs-simulation comparisons:
A, A11, Ab=Ab1+Abp (no mixed states), Abb. Theory is recalculated using
beta=tau/tsw and the self-consistent passive-c mean-field correction.
Mode 4 uses an explicitly labeled Poisson surrogate, NOT periodic theory.
Modes 1/3/4 use active/passive nomenclature of the 9-state theory.
Mode 2 has two active arms, so the nine states are defined left/right;
active/passive theory labels are intentionally not used for model 2.

Legacy observational width options (--size, --multiplier) remain accepted,
but are ignored by collision-event classification. The only geometric
collision tolerance is --dc. Grid mode defaults to one legacy size value
(0.015) instead of running multiple physically identical simulations.

Examples:
    python cohesin_simulator_poisson_4populations.py --grid \
        --model 4 --dc-grid 0.001,0.003,0.01 \
        --tsw-grid 0.005,0.01,0.02 --multiplier 2 --n-all-c 100001 --n-ch 200

    python cohesin_simulator_poisson_4populations.py --grid \
        --model 1 --dc-grid 0.001,0.003,0.01 \
        --multiplier-grid 1,2,3 --n-all-c 100001 --n-ch 200

Preserves the original collision/movement algorithm and time grid. Population
classification now uses b-obstacle IDs recorded when the movement attempt is
actually blocked; it does NOT infer collisions from distance-to-obstacle masks.
The recorded b-block persists through direction switching, until that obstacle
unloads or the arm successfully moves (or hits a different blocker).
A, A_bb, A_b, A_00 form a disjoint partition at every sampled snapshot.
The optional --size / --multiplier are retained for CLI compatibility but
DO NOT affect classification, because only --dc determines collision geometry. For mode 4, fixed-interval direction switching uses the SAME RNG
initialization/order as the original, allowing direct seed-by-seed validation.

Acceleration: Numba JIT, event-based active object index, and direct updating of
arm lengths (no per-step N_all-sized array allocations). Numba is required:
    pip install numba

For models 3 and 5 switch events are generated beforehand. This retains the
stochastic switching law but consumes random numbers in a different order
than the original implementation; same seed does not imply identical paths.
"""

import sys
import time
import os
import json
import argparse
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.linalg import expm
from scipy.optimize import brentq

try:
    from numba import njit, types
    from numba.typed import List
except ImportError as exc:
    raise ImportError("Install numba to run this accelerated version: pip install numba") from exc

# Identical physical and numerical settings to the supplied script.
model = 4

dt = 0.001
dc = 0.015

tsw = 0.01
tswMin = 0.01

v = 1
tau = 1

sample_dt = 0.1 * tau
sample_every = max(1, round(sample_dt / dt))

n_all_c = 1000000
n_ch = 50
t_full = round(n_all_c * tau / n_ch)

# Only affects the frequency of progress reports, not the simulated dynamics.
NUMBA_CHUNK_STEPS = 4000
GAMMA_FOR_PLOT, GAMMA_B_FOR_PLOT, ALPHA_FOR_PLOT = 2.7, 4.0, 1.2
PASSIVE_C_COLLISIONS = True
THEORY_COMPARISON_PLOTS = True
STATE_CLASSIFICATION_VERSION = "sv2"  # direction read per sampled cohesin


def format_duration(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def sample_switch_interval(size):
    if model == 3:
        return np.random.exponential(scale=tsw, size=size)
    if model == 4:
        return np.full(size, tsw, dtype=float)
    if model == 5:
        return tswMin + np.random.exponential(scale=tsw - tswMin, size=size)
    raise ValueError("Switch intervals exist only for modes 3, 4, 5")


def _prepare_switch_events(model_id, start_c, end_c, initial_next, end_t):
    """All stochastic switch times, only for modes 3 and 5.

    Switching intervals are IID as in the original code, but the order in
    which numpy draws them differs. Mode 4 uses a deterministic clock inside
    the compiled loop and does not use this function.
    """
    if model_id not in (3, 5):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    times = []
    indices = []
    for i in range(len(start_c)):
        limit = min(end_c[i], end_t)
        next_t = initial_next[i]
        while next_t <= limit:
            times.append(next_t)
            indices.append(i)
            if model_id == 3:
                next_t += np.random.exponential(scale=tsw)
            else:
                next_t += tswMin + np.random.exponential(scale=tsw - tswMin)

    if not times:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    order = np.argsort(np.asarray(times, dtype=np.float64))
    return np.asarray(times, dtype=np.float64)[order], np.asarray(indices, dtype=np.int64)[order]


@njit(cache=False)
def _simulate_chunk(time_values, first_step, step_sample_every, half_time,
                    model_id, dc_local, dl, tsw_local, norm, passive_c_enabled,
                    n_cohesins, start, end, pos, direction,
                    start_order, end_order, sw_times, sw_ids, next_sw,
                    start_ptr, end_ptr, sw_ptr, active_ids, is_active, n_active,
                    left_len, right_len, switch_counts, left_blocker_id, right_blocker_id,
                    collect_raw, only_bb, a_out, abb_out, ab_out, a00_out, state_out,
                    r_out, l_out, ids_out,
                    total_snapshots, total_b_collision_events, total_passive_c_events):
    """Simulate original movement; track actual b blockers of each leg.

    left_blocker_id/right_blocker_id store confirmed blocker object IDs,
    including b barriers and other cohesins c. -1 means no current blocker.
    Registered blockage survives a direction switch (passive-leg memory),
    but is cleared upon obstacle unloading or when the arm can move again.
    When an advancing c tip contacts a passive c tip, the hit is registered
    on BOTH cohesins (on their appropriate arms). Existing b blockers take
    precedence over additional c contacts, since each arm stores only one ID.
    """
    n_objects = len(start)

    for local_step in range(len(time_values)):
        t = time_values[local_step]

        # Process starts FIRST, then ends: accommodates objects born and gone
        # between two time steps without spuriously leaving them active.
        while start_ptr < n_objects and start[start_order[start_ptr]] <= t:
            ident = start_order[start_ptr]
            # Insert by the immutable loading coordinate (the original sort key).
            lo, hi = 0, n_active
            while lo < hi:
                mid = (lo + hi) // 2
                if pos[active_ids[mid]] <= pos[ident]:
                    lo = mid + 1
                else:
                    hi = mid
            for q in range(n_active, lo, -1):
                active_ids[q] = active_ids[q - 1]
            active_ids[lo] = ident
            n_active += 1
            is_active[ident] = True
            start_ptr += 1

        while end_ptr < n_objects and end[end_order[end_ptr]] <= t:
            ident = end_order[end_ptr]
            if is_active[ident]:
                q = 0
                while q < n_active and active_ids[q] != ident:
                    q += 1
                if q < n_active:
                    for k in range(q, n_active - 1):
                        active_ids[k] = active_ids[k + 1]
                    n_active -= 1
                is_active[ident] = False
            end_ptr += 1

        # Mode 4: retain the original deterministic clock, including all
        # switches due at t even if several events fit into one dt.
        if model_id == 4:
            for k in range(n_active):
                ident = active_ids[k]
                if ident < n_cohesins:
                    while next_sw[ident] <= t:
                        direction[ident] = 1 - direction[ident]
                        switch_counts[ident] += 1
                        next_sw[ident] += tsw_local
        elif model_id == 3 or model_id == 5:
            while sw_ptr < len(sw_times) and sw_times[sw_ptr] <= t:
                ident = sw_ids[sw_ptr]
                if is_active[ident]:
                    direction[ident] = 1 - direction[ident]
                    switch_counts[ident] += 1
                sw_ptr += 1

        # A moving tip can encounter a barrier b or a cohesin c. If it hits
        # the PASSIVE tip of c, register the same physical collision on the
        # passive tip as well. No additional spatial search/proximity event.
        #
        # Geometric approach to a neighbor on the right: our moving RIGHT
        # tip meets the neighbor's LEFT tip. Conversely our moving LEFT tip
        # meets the RIGHT tip of a neighbor on the left.
        for i in range(n_active):
            ident = active_ids[i]
            if ident >= n_cohesins:
                continue

            llast = left_blocker_id[ident]
            if llast >= 0 and not is_active[llast]:
                left_blocker_id[ident] = -1
            rlast = right_blocker_id[ident]
            if rlast >= 0 and not is_active[rlast]:
                right_blocker_id[ident] = -1

            d = direction[ident]
            if d == 0 or d == 2:
                blocked_by = -1
                hit_left_tip = False
                if i < n_active - 1:
                    tip = pos[ident] + right_len[ident]
                    for j in range(i + 1, n_active):
                        other = active_ids[j]
                        other_l = pos[other] - left_len[other]
                        other_r = pos[other] + right_len[other]
                        # Preserve the old movement rule for an exact paired
                        # control run; identify which of the two tips was hit.
                        if other_l > tip - dc_local and other_l < tip + dc_local:
                            blocked_by = other
                            hit_left_tip = True
                            break
                        if other_r > tip - dc_local and other_r < tip + dc_local:
                            blocked_by = other
                            hit_left_tip = False
                            break
                if blocked_by < 0:
                    right_len[ident] += dl
                    right_blocker_id[ident] = -1
                else:
                    if blocked_by >= n_cohesins and right_blocker_id[ident] != blocked_by:
                        total_b_collision_events += 1
                    right_blocker_id[ident] = blocked_by
                    # other is a cohesin; its LEFT arm is passive when it is
                    # actively extruding to the RIGHT (direction==0).
                    if (passive_c_enabled and blocked_by < n_cohesins and
                            hit_left_tip and direction[blocked_by] == 0):
                        previous = left_blocker_id[blocked_by]
                        if previous < 0 or not is_active[previous]:
                            left_blocker_id[blocked_by] = ident
                            total_passive_c_events += 1

            if d == 1 or d == 2:
                blocked_by = -1
                hit_right_tip = False
                if i > 0:
                    tip = pos[ident] - left_len[ident]
                    for j in range(i - 1, -1, -1):
                        other = active_ids[j]
                        other_r = pos[other] + right_len[other]
                        other_l = pos[other] - left_len[other]
                        if other_r < tip + dc_local and other_r > tip - dc_local:
                            blocked_by = other
                            hit_right_tip = True
                            break
                        if other_l < tip + dc_local and other_l > tip - dc_local:
                            blocked_by = other
                            hit_right_tip = False
                            break
                if blocked_by < 0:
                    left_len[ident] += dl
                    left_blocker_id[ident] = -1
                else:
                    if blocked_by >= n_cohesins and left_blocker_id[ident] != blocked_by:
                        total_b_collision_events += 1
                    left_blocker_id[ident] = blocked_by
                    # other is a cohesin; its RIGHT arm is passive when it
                    # actively extrudes to the LEFT (direction==1).
                    if (passive_c_enabled and blocked_by < n_cohesins and
                            hit_right_tip and direction[blocked_by] == 1):
                        previous = right_blocker_id[blocked_by]
                        if previous < 0 or not is_active[previous]:
                            right_blocker_id[blocked_by] = ident
                            total_passive_c_events += 1

        global_step = first_step + local_step + 1
        if t > half_time and global_step % step_sample_every == 0:
            for k in range(n_active):
                ident = active_ids[k]
                if ident >= n_cohesins:
                    continue
                total_snapshots += 1
                length = (left_len[ident] + right_len[ident]) / norm
                if collect_raw:
                    a_out.append(length)

                # A collision state is defined by a confirmed, still-valid
                # blocker, never by the proximity of a loop end to an obstacle.
                lb = left_blocker_id[ident]
                rb = right_blocker_id[ident]
                if lb >= 0 and not is_active[lb]:
                    lb = -1
                    left_blocker_id[ident] = -1
                if rb >= 0 and not is_active[rb]:
                    rb = -1
                    right_blocker_id[ident] = -1

                # IMPORTANT: the active/passive ordering belongs to THIS
                # cohesin, not the last cohesin iterated in the movement loop.
                d = direction[ident]
                # 0=free, 1=b barrier, 2=other cohesin c
                lt = 0 if lb < 0 else (1 if lb >= n_cohesins else 2)
                rt = 0 if rb < 0 else (1 if rb >= n_cohesins else 2)

                # One physical b object cannot count as two distinct b barriers.
                # The following exceptional degeneracy, if it occurs, is assigned
                # to one b contact (the passive arm in one-sided models).
                if lt == 1 and rt == 1 and lb == rb:
                    if d == 0:
                        rt = 0
                    else:
                        lt = 0

                is_bb = lt == 1 and rt == 1
                has_b = lt == 1 or rt == 1

                # Nine disjoint states. Model 2 has two active legs and uses
                # ordered (left,right). Other models use (active,passive).
                if d == 2:
                    state_index = 3 * lt + rt
                elif d == 0:
                    state_index = 3 * rt + lt
                else:
                    state_index = 3 * lt + rt

                if collect_raw:
                    state_out[state_index].append(length)
                    if is_bb:
                        abb_out.append(length)
                    elif has_b:
                        ab_out.append(length)
                    else:
                        a00_out.append(length)
                if (not only_bb) or is_bb:
                    r_out.append(right_len[ident] / norm)
                    l_out.append(left_len[ident] / norm)
                    ids_out.append(ident)

    return (start_ptr, end_ptr, sw_ptr, n_active, total_snapshots,
            total_b_collision_events, total_passive_c_events)


# Analytical expressions provided by the user. IMPORTANT: model-4 simulations use
# the BARE cohesin density gamma=2.7; the value 3.7 in the theoretical formulas
# includes the dynamic cohesin collision correction: gamma_c ~= gamma + gamma_dyn.
# They MUST NOT be set equal by changing the simulation gamma.
# xi denotes physical separation in Mb, with xi = 0.431*x and x=L/lp.
# Exact nine-state labels (not independent of the aggregate four populations).
# Index = 3 * first_leg_type + second_leg_type, with 0=free, 1=b, 2=c.
# For one-sided modes first=active, second=passive.
STATE_NAMES_ACTIVE_PASSIVE = (
    "A11", "Ab1", "Ac1", "Abp", "Abb", "Abc", "Acp", "Acb", "Acc"
)
# In model 2 both arms move, so first=left, second=right.
STATE_NAMES_LEFT_RIGHT = (
    "A_L0_R0", "A_L0_Rb", "A_L0_Rc",
    "A_Lb_R0", "A_Lb_Rb", "A_Lb_Rc",
    "A_Lc_R0", "A_Lc_Rb", "A_Lc_Rc"
)


def state_names_for_model(model_id):
    return STATE_NAMES_LEFT_RIGHT if model_id == 2 else STATE_NAMES_ACTIVE_PASSIVE


LP_MB = 0.431
THEORY_GAMMA_C = 3.7
THEORY_GAMMA_B = 4.0
THEORY_ALPHA = 1.2
THEORY_COEFS = {
    "A": np.array([
        0.0000156304 * 630940.0,
       -0.0000156304 * 474889.0,
       -0.0000156304 * 12100.9,
    ], dtype=np.float64),
    "A_bb": np.array([
        8.0,
        2.26788e-7 * 4.14125e7,
       -2.26788e-7 * 4.8805e7,
        2.26788e-7 * 7.39246e6,
    ], dtype=np.float64),
}
THEORY_RATES = {
    "A": np.array([7.91138, 7.91138 + 19.7924,
                   7.91138 + 13.4423]),
    "A_bb": np.array([303.03, 7.91138, 7.91138 + 19.7924,
                      7.91138 + 13.4423]),
}
THEORY_AREAS_XI = {
    key: float(np.sum(THEORY_COEFS[key] / THEORY_RATES[key]))
    for key in THEORY_COEFS
}


def theory_pdf_x(x, name, variant="full"):
    """PDF with respect to dimensionless x=L/lp, normalized on [0, inf)."""
    x = np.asarray(x, dtype=np.float64)
    xi = LP_MB * x
    coefs = THEORY_COEFS[name]
    rates = THEORY_RATES[name]
    if name == "A_bb" and variant == "no_short":
        # Drop only the explicit short-range term 8 exp(-303.03 xi),
        # while keeping the normalization of the full reference curve.
        coefs = coefs[1:]
        rates = rates[1:]
        area = THEORY_AREAS_XI["A_bb"]
    else:
        area = THEORY_AREAS_XI[name]
    raw_xi = np.sum(coefs[:, None]
                    * np.exp(-rates[:, None] * xi.reshape(1, -1)),
                    axis=0)
    return LP_MB * raw_xi / area


def theory_bin_masses(bins, name):
    """Exact integrals, not midpoint approximations, for nonuniform bins."""
    a = LP_MB * np.asarray(bins[:-1])
    b = LP_MB * np.asarray(bins[1:])
    out = np.zeros(len(a), dtype=np.float64)
    for coeff, rate in zip(THEORY_COEFS[name], THEORY_RATES[name]):
        # exp(-r*a) - exp(-r*b), stable for narrow bins
        out += coeff * np.exp(-rate * a) * (-np.expm1(-rate * (b - a))) / rate
    return out / THEORY_AREAS_XI[name]


def histogram_bins():
    """Mathematica: Join[Range[.0001,.01,.002],Range[.02,.8,.01]]."""
    fine = .0001 + .002 * np.arange(5, dtype=float)  # .0001,...,.0081
    coarse = .02 + .01 * np.arange(79, dtype=float)  # .02,...,.80
    return np.r_[fine, coarse]


def histogram_data(samples, name):
    bins = histogram_bins()
    counts, edges = np.histogram(samples, bins=bins)
    if len(samples) == 0:
        return edges, counts, np.zeros(len(counts)), float("nan"), float("nan")
    # IMPORTANT: denominator includes all samples, even outside plotted range.
    empirical_mass = counts / len(samples)
    density = empirical_mass / np.diff(edges)
    expected_mass = theory_bin_masses(edges, name)
    nmae = (float(np.abs(empirical_mass - expected_mass).sum()
             / expected_mass.sum()))
    observed_window = empirical_mass.sum()
    expected_window = expected_mass.sum()
    nmae_shape = (float(np.abs(empirical_mass / observed_window
                               - expected_mass / expected_window).sum())
                  if observed_window > 0 else float("nan"))
    return edges, counts, density, nmae, nmae_shape


def number_tag(gamma, gamma_b, alpha, width, run_id=None):
    tag = (f"collision_pc{int(PASSIVE_C_COLLISIONS)}_{STATE_CLASSIFICATION_VERSION}_m{model}_n{n_all_c}_nch{n_ch}"
           f"_dc{dc:g}_size{width:g}_tsw{tsw:g}"
           f"_g{gamma:g}_gb{gamma_b:g}_alpha{alpha:g}")
    return tag + ("_" + str(run_id) if run_id is not None else "_run" + uuid4().hex[:8])


def atomic_csv_x(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    pd.DataFrame({"x": samples}).to_csv(tmp, index=False, float_format="%.17g")
    os.replace(tmp, path)


def export_population_histograms(output, tag, arrays):
    """Save counts and global/within-population PDFs for loop lengths.

    Bin edges: 1 kb below 10 kb, 10 kb thereafter, at least through 800 kb.
    If any loops exceed 800 kb, the last edge extends to contain all samples.
    Global PDFs are divided by N_A; their subgroup integrals are the
    subgroup proportions. Conditional PDFs are divided by each subgroup N.
    """
    n_total = len(arrays["A"])
    if n_total == 0:
        raise ValueError("No sampled loops; increase --n-all-c or sampling frequency")
    max_kb = max(800.0, float(np.max(arrays["A"])) * LP_MB * 1000)
    top_kb = 10 * int(np.ceil(max_kb / 10.0))
    bins_kb = np.r_[np.arange(0.0, 11.0, 1.0),
                    np.arange(20.0, top_kb + 10.0, 10.0)]
    widths = np.diff(bins_kb)
    frame = pd.DataFrame({
        "left_kb": bins_kb[:-1], "right_kb": bins_kb[1:],
        "center_kb": (bins_kb[:-1] + bins_kb[1:]) / 2,
        "width_kb": widths,
        "left_x": bins_kb[:-1] / (LP_MB * 1000),
        "right_x": bins_kb[1:] / (LP_MB * 1000),
        "center_x": (bins_kb[:-1] + bins_kb[1:]) / (2 * LP_MB * 1000),
        "width_x": widths / (LP_MB * 1000),
    })
    for name in arrays:
        counts = np.histogram(arrays[name] * LP_MB * 1000, bins=bins_kb)[0]
        frame[f"count_{name}"] = counts
        frame[f"global_pdf_{name}_per_kb"] = counts / (n_total * widths)
        frame[f"conditional_pdf_{name}_per_kb"] = (
            counts / (len(arrays[name]) * widths)
            if len(arrays[name]) else np.full(len(counts), np.nan)
        )
        frame[f"global_pdf_{name}_per_x"] = (
            frame[f"global_pdf_{name}_per_kb"] * LP_MB * 1000
        )
        frame[f"conditional_pdf_{name}_per_x"] = (
            frame[f"conditional_pdf_{name}_per_kb"] * LP_MB * 1000
        )
    assert np.array_equal(frame["count_A"].to_numpy(),
                          frame[["count_A_bb", "count_A_b", "count_A_00"]]
                          .sum(axis=1).to_numpy())
    state_names = [name for name in arrays if name not in ("A", "A_bb", "A_b", "A_00")]
    if state_names:
        assert len(state_names) == 9
        assert np.array_equal(frame["count_A"].to_numpy(),
                              frame[[f"count_{name}" for name in state_names]]
                              .sum(axis=1).to_numpy())
    out_file = output / f"distributions_4pop_{tag}.csv"
    atomic_csv_df(out_file, frame)
    print("Saved binned distributions:", out_file)
    return out_file



# ============================================================================
# State-resolved mean-field theory used in cohesin_poisson_beta_functions.wl.
# All transition rates are per cohesin lifetime tau; x = L/(v*tau).
# Theory names are ordered (active, passive) for Poisson switching, or
# (left, right) for symmetric two-sided extrusion.
# This theoretical machinery is independent of the original, fixed-gamma_c=3.7
# reference (which is not appropriate for varying Poisson switching rates).
# ============================================================================
THEORY_NINE_STATES = ("ff", "fb", "fc", "bf", "cf", "bb", "bc", "cb", "cc")
THEORY_NINE_INDEX = {name: i for i, name in enumerate(THEORY_NINE_STATES)}
THEORY_COMPARISON_NAMES = ("A", "A11", "Ab", "Abb")

# Explicit name-to-state mapping: the simulation output order differs from
# THEORY_NINE_STATES at indices 4, 5, and 6. Never zip those orders.
SIM_STATE_CODES = {
    "A11": "ff", "Ab1": "fb", "Ac1": "fc",
    "Abp": "bf", "Abb": "bb", "Abc": "bc",
    "Acp": "cf", "Acb": "cb", "Acc": "cc",
    "A_L0_R0": "ff", "A_L0_Rb": "fb", "A_L0_Rc": "fc",
    "A_Lb_R0": "bf", "A_Lb_Rb": "bb", "A_Lb_Rc": "bc",
    "A_Lc_R0": "cf", "A_Lc_Rb": "cb", "A_Lc_Rc": "cc",
}


def _latex_state_name(name):
    if name.startswith("A_L"):
        return r"$A_{" + SIM_STATE_CODES[name].replace("f", "0") + r"}$"
    if name == "A11":
        return r"$A_{11}$"
    return r"$A_{" + name[1:] + r"}$"


def _moving_fractions_poisson(beta, gamma, gamma_b, alpha):
    """Stationary fractions for the same mean-field equations as the WM file."""
    c = 1.0 + alpha
    a = gamma_b * (c + beta) / (c * (c + 2.0 * beta))
    p = gamma_b * beta / (c * (c + 2.0 * beta))

    def passive_free(j):
        return (2.0 + j * (beta * (1.0 + a) - (2.0 + beta) * p)) / (
            2.0 + beta + gamma * j / 2.0)

    def residual(j):
        return ((gamma + gamma * j / 2.0) * j
                - (2.0 + beta) * (1.0 - (1.0 + a) * j)
                + beta * (1.0 - p * j - passive_free(j)))

    J = brentq(residual, 0.0, 1.0, xtol=1e-14)
    P = passive_free(J)
    h = gamma * J / 2.0
    return J, P, h, gamma + h


def _moving_fraction_two_sided(gamma, gamma_b, alpha):
    """J = 1 / (1+gb/[2(1+alpha)]+gamma*(1+J)/4)."""
    def residual(j):
        return j * (1.0 + gamma_b / (2.0 * (1.0 + alpha))
                    + gamma * (1.0 + j) / 4.0) - 1.0

    J = brentq(residual, 0.0, 1.0, xtol=1e-14)
    return J, gamma * (1.0 + J)


def switching_mean_field_theory(model_id, switch_tsw, gamma, gamma_b, alpha):
    """Build analytic length PDFs from a 9-state linear transport model.

    Models 1/3: original Poisson-switching theory with beta=0 or tau/tsw.
    Model 2: symmetric two-sided model, each free arm advances at speed v/2.
    Model 4: Poisson surrogate beta=tau/tsw ONLY (not exact for periodic switches).

    Return rows in state order from THEORY_NINE_STATES. The density of a
    population is row @ exp(M*x) @ e0. Divide by row @ (-M)^-1 @ e0
    for the PDF conditional on membership in that state/population.
    """
    if model_id == 2:
        beta = None
        J, gc = _moving_fraction_two_sided(gamma, gamma_b, alpha)
        P = J
        h = gamma * J / 2.0
        kind = "symmetric two-sided mean-field"
    elif model_id in (1, 3, 4):
        beta = 0.0 if model_id == 1 else float(tau / switch_tsw)
        J, P, h, gc = _moving_fractions_poisson(beta, gamma, gamma_b, alpha)
        kind = {
            1: "one-sided no-switching mean-field",
            3: "Poisson-switching mean-field",
            4: "Poisson surrogate for PERIODIC switching (not exact)",
        }[model_id]
    else:
        raise ValueError("Theory comparison supports models 1, 2, 3 and 4")

    Q = -np.eye(9, dtype=np.float64)  # cohesin dissociation at rate 1

    def add_transition(source, target, rate):
        if source == target or rate == 0.0:
            return
        i, j = THEORY_NINE_INDEX[source], THEORY_NINE_INDEX[target]
        Q[j, i] += rate
        Q[i, i] -= rate

    for state in THEORY_NINE_STATES:
        first, second = state
        if model_id == 2:
            # Both legs are active with half-speed; encounters per arm are
            # gamma_b/2 and gamma_c/2. There is no directional switching.
            if first == "f":
                add_transition(state, "b" + second, gamma_b / 2.0)
                add_transition(state, "c" + second, gc / 2.0)
            if second == "f":
                add_transition(state, first + "b", gamma_b / 2.0)
                add_transition(state, first + "c", gc / 2.0)
        else:
            # First = active, second = passive, with extra collisions from
            # other moving cohesins onto the passive free leg.
            if first == "f":
                add_transition(state, "b" + second, gamma_b)
                add_transition(state, "c" + second, gc)
            if second == "f":
                add_transition(state, first + "c", h)
            if beta > 0:
                add_transition(state, second + first, beta)
        if first == "b":
            add_transition(state, "f" + second, alpha)
        if second == "b":
            add_transition(state, first + "f", alpha)
        if first == "c":
            add_transition(state, "f" + second, 1.0)
        if second == "c":
            add_transition(state, first + "f", 1.0)

    if model_id == 2:
        moving = np.asarray([i for i, s in enumerate(THEORY_NINE_STATES)
                             if "f" in s], dtype=np.int64)
        blocked = np.asarray([i for i, s in enumerate(THEORY_NINE_STATES)
                              if "f" not in s], dtype=np.int64)
        velocities = np.asarray([THEORY_NINE_STATES[i].count("f") / 2.0
                                 for i in moving])
    else:
        moving = np.asarray([0, 1, 2], dtype=np.int64)
        blocked = np.asarray([3, 4, 5, 6, 7, 8], dtype=np.int64)
        velocities = np.ones(len(moving))

    qm = Q[np.ix_(moving, moving)]
    qb = Q[np.ix_(moving, blocked)]
    bm = Q[np.ix_(blocked, moving)]
    bb = Q[np.ix_(blocked, blocked)]
    S = -np.linalg.solve(bb, bm)
    M = (qm + qb @ S) / velocities[:, None]
    rows9 = np.zeros((9, len(moving)), dtype=float)
    rows9[moving, :] = np.eye(len(moving))
    rows9[blocked, :] = S
    e0 = np.zeros(len(moving), dtype=float)
    e0[0] = 1.0  # probability flux of newly loaded, fully free cohesins
    i_moving = np.linalg.solve(-M, e0)

    # Ab excludes mixed bc/cb states, exactly as in the Mathematica Abnorm.
    # These index groups are valid for both active/passive and left/right theory.
    rows = {
        "A": rows9.sum(axis=0),
        "A11": rows9[0],
        "Ab": rows9[1] + rows9[3],
        "Abb": rows9[5],
    }
    # Each of the nine mutually exclusive states (including all mixed states).
    # State-specific PDFs must use these exact rows, not an aggregate A_b.
    for sim_name, state_code in SIM_STATE_CODES.items():
        rows[sim_name] = rows9[THEORY_NINE_INDEX[state_code]]
    weights = {name: float(row @ i_moving) for name, row in rows.items()}
    mean_x = float(rows["A"] @ np.linalg.solve(-M, i_moving))

    if not np.isclose(weights["A"], 1.0, atol=5e-10):
        raise ArithmeticError(f"Theory A not normalized: {weights['A']}")
    moving_fraction_integrated = float(velocities @ i_moving)
    if not np.isclose(moving_fraction_integrated, J, atol=5e-10):
        raise ArithmeticError("Moving-fraction self-consistency failed")
    if not np.isclose(mean_x, J, atol=5e-10):
        raise ArithmeticError("Mean length differs from mobile fraction")
    if np.any(np.real(np.linalg.eigvals(M)) >= -1e-10):
        raise ArithmeticError("Theory matrix is not stable")

    return dict(model=model_id, kind=kind, beta=beta, J=J, P=P, h=h,
                gamma_c_active=gc, mean_x=mean_x, M=M, e0=e0,
                rows=rows, weights=weights)


def theoretical_density_and_bin_masses(theory, bin_edges_x, dense_x, names=None):
    """Compute smooth conditional PDFs and EXACT bin-averaged theory, in x.

    Bin masses integrate the full matrix exponential, not theory evaluated
    at interval midpoints. The bins are nonuniform (1 kb, then 10 kb).
    """
    M = theory["M"]
    e0 = theory["e0"]
    unit = np.eye(M.shape[0])
    dense_moving = np.stack([expm(M * xx) @ e0 for xx in dense_x])
    int_at_edges = np.stack([
        np.linalg.solve(M, (expm(M * xx) - unit) @ e0)
        for xx in bin_edges_x
    ])
    result = {}
    if names is None:
        names = THEORY_COMPARISON_NAMES
    for name in names:
        weight = theory["weights"][name]
        if weight <= 1e-12:
            result[name] = None
            continue
        row = theory["rows"][name]
        smooth = np.maximum(0., (dense_moving @ row) / weight)
        cdf = (int_at_edges @ row) / weight
        bin_masses = np.diff(cdf)
        # Floating-point cancellations may create tiny negative masses, but
        # no physical bin has negative probability.
        bin_masses[np.logical_and(bin_masses < 0., bin_masses > -1e-10)] = 0.
        if np.any(bin_masses < -1e-10):
            raise ArithmeticError(f"Negative theory bin mass for {name}")
        result[name] = dict(smooth=smooth, bin_masses=bin_masses,
                            bin_pdf=bin_masses / np.diff(bin_edges_x))
    return result


def _theory_simulation_samples(model_id, state_arrays, all_samples):
    """Match four EXACT populations to the theory, never use coarse A_b."""
    if model_id == 2:
        single = ("A_L0_Rb", "A_Lb_R0")
        free = "A_L0_R0"
        double = "A_Lb_Rb"
    else:
        single = ("Ab1", "Abp")
        free = "A11"
        double = "Abb"
    return {
        "A": all_samples,
        "A11": state_arrays[free],
        "Ab": np.concatenate((state_arrays[single[0]], state_arrays[single[1]])),
        "Abb": state_arrays[double],
    }


def plot_theory_vs_simulation_four_states(
        model_id, switch_tsw, gamma, gamma_b, alpha,
        state_arrays, all_samples, histogram_csv, output_dir, tag,
        save_individual=True):
    """Plot A, A11, Ab=Ab1+Abp, Abb; save exact-bin theory and metrics.

    All curves use the conditional PDF of x=L/lp within the respective
    population. For A, the population is all loops, so the same rule applies.
    """
    theory = switching_mean_field_theory(model_id, switch_tsw, gamma, gamma_b, alpha)
    samples = _theory_simulation_samples(model_id, state_arrays, all_samples)
    df = pd.read_csv(histogram_csv)
    edges = np.r_[df["left_x"].to_numpy(dtype=float), df["right_x"].iloc[-1]]
    widths = np.diff(edges)
    xmin, xmax = 0.004, 1.75
    dense = np.geomspace(xmin, xmax, 850)
    reference = theoretical_density_and_bin_masses(theory, edges, dense)

    descriptions = {
        "A": r"$A$  (all loops)",
        "A11": r"$A_{11}$  (both arms free)",
        "Ab": r"$A_b=A_{b1}+A_{bp}$  (one b, other free)",
        "Abb": r"$A_{bb}$  (both arms b-blocked)",
    }
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.6), dpi=140)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"theory": theory["kind"], "model": int(model_id),
              "beta": (float(theory["beta"]) if theory["beta"] is not None else None),
              "gamma": float(gamma), "gamma_b": float(gamma_b), "alpha": float(alpha),
              "gamma_c_active": float(theory["gamma_c_active"]),
              "J": float(theory["J"]), "P": float(theory["P"]),
              "h": float(theory["h"]), "mean_x_theory": float(theory["mean_x"]),
              "mean_x_simulation": float(np.mean(all_samples)),
              "theory_matrices": "9-state Markov generator; Schur elimination",
              "PDF_normalization": "each of A, A11, Ab, Abb individually integrates to 1",
              "Ab_definition": "Ab1+Abp (excluding mixed bc and cb)",
              "populations": {}}
    theory_csv = df[["left_x", "right_x", "center_x", "width_x",
                     "left_kb", "right_kb"]].copy()
    plot_output = output_dir / f"Plot_Theory_vs_Simulation_4states_{tag}.png"

    for ax, name in zip(axes.ravel(), THEORY_COMPARISON_NAMES):
        values = samples[name]
        counts = np.histogram(values, bins=edges)[0]
        observed_pdf = (counts / (len(values) * widths)
                        if len(values) else np.full(len(widths), np.nan))
        theory_item = reference[name]
        observed_share = len(values) / len(all_samples)
        theory_share = theory["weights"][name]
        if len(values):
            ax.stairs(observed_pdf, edges, baseline=0, fill=True,
                      color="#80E5EC", alpha=0.58, edgecolor="#188D99",
                      linewidth=0.65, label="Simulation (bin PDF)")
        else:
            ax.text(0.5, 0.43, "No simulated samples", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
        if theory_item is not None:
            ax.plot(dense, theory_item["smooth"], color="#C46D31",
                    linewidth=2.1, label="Theory (continuous)")
            ax.stairs(theory_item["bin_pdf"], edges, color="#314D91",
                      linewidth=1.35, linestyle="--",
                      label="Theory (same bins)")
            nmae = (float(np.sum(np.abs(counts / len(values)
                                          - theory_item["bin_masses"])))
                    if len(values) else None)
        else:
            nmae = None
            ax.text(0.5, 0.5, "Theory state absent", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10)
        title = (descriptions[name] + f"\nN={len(values):,}; "
                 + f"fraction sim {observed_share:.3f}, theory {theory_share:.3f}")
        ax.set_title(title, fontsize=11)
        ax.set(xlabel=r"Loop length  $x=L/l_p$", ylabel=r"PDF  ($1/x$)")
        ax.set_xlim(xmin, xmax)
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8, frameon=False)

        theory_csv[f"sim_count_{name}"] = counts
        theory_csv[f"sim_pdf_{name}_per_x"] = observed_pdf
        theory_csv[f"theory_bin_pdf_{name}_per_x"] = (
            theory_item["bin_pdf"] if theory_item is not None else np.nan)
        theory_csv[f"theory_bin_mass_{name}"] = (
            theory_item["bin_masses"] if theory_item is not None else np.nan)
        report["populations"][name] = {
            "n": int(len(values)), "simulated_fraction": float(observed_share),
            "theoretical_fraction": float(theory_share),
            "absolute_bin_mass_error": nmae,
            "mean_x_simulation": float(np.mean(values)) if len(values) else None,
        }

        if save_individual:
            # Render each single-panel figure directly to keep labels readable.
            f1, a1 = plt.subplots(figsize=(7.3, 5.0), dpi=145)
            if len(values):
                a1.stairs(observed_pdf, edges, baseline=0, fill=True,
                          color="#80E5EC", alpha=.58, edgecolor="#188D99",
                          linewidth=.65, label="Simulation")
            else:
                a1.text(0.5, 0.43, "No simulated samples",
                        transform=a1.transAxes, ha="center", va="center")
            if theory_item is not None:
                a1.plot(dense, theory_item["smooth"], color="#C46D31",
                        linewidth=2., label="Theory")
                a1.stairs(theory_item["bin_pdf"], edges, color="#314D91",
                          linewidth=1.3, linestyle="--", label="Theory (bin average)")
            a1.set(xlabel=r"Loop length  $x=L/l_p$", ylabel="PDF (per x)")
            a1.set_xlim(xmin, xmax)
            a1.set_xscale("log")
            a1.set_ylim(bottom=0)
            a1.set_title(title, fontsize=11)
            a1.grid(alpha=.18)
            if a1.get_legend_handles_labels()[0]:
                a1.legend(frameon=False, fontsize=9)
            f1.tight_layout()
            f1.savefig(output_dir / f"Plot_Theory_vs_Simulation_{name}_{tag}.png",
                       dpi=145, bbox_inches="tight")
            plt.close(f1)

    label = theory["kind"]
    if model_id == 4:
        label += "\nWARNING: periodic switching is NOT Poisson switching"
    fig.suptitle(
        f"Simulation versus theory | {label} | "
        + (f"beta={theory['beta']:g}" if theory["beta"] is not None else "two-sided")
        + f" | gamma={gamma:g}, gb={gamma_b:g}, alpha={alpha:g}, dc={dc:g}",
        y=1.01, fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_output, dpi=145, bbox_inches="tight")
    plt.close(fig)
    theory_csv_file = output_dir / f"Theory_vs_Simulation_4states_bins_{tag}.csv"
    atomic_csv_df(theory_csv_file, theory_csv)
    report_file = output_dir / f"Theory_vs_Simulation_4states_stats_{tag}.json"
    _write_json(report_file, report)
    print("Saved theory comparisons:", plot_output)
    print("Saved exact-bin theory:", theory_csv_file)
    print("Saved theory population fractions:", report_file)
    if model_id == 4:
        print("WARNING: fixed-interval switches compared with a Poisson SURROGATE; "
              "this is NOT the exact periodic-switching theory.")
    return plot_output, theory_csv_file, report_file



def plot_theory_vs_simulation_nine_states(
        model_id, switch_tsw, gamma, gamma_b, alpha,
        histogram_csv, output_dir=None, tag=None):
    """Compare ALL nine exclusive b/c collision classes to the 9-state theory.

    Works from distributions_all_states_*.csv alone, so an already completed
    million-event simulation DOES NOT have to be re-run.

    Figures: conditional PDF 3x3, globally normalized PDF 3x3, class fractions,
    active/passive (or left/right) 3x3 probability matrices.
    Data: state-by-state CSV, theory bin PDF CSV, summary JSON.
    """
    histogram_csv = Path(histogram_csv)
    output_dir = Path(output_dir) if output_dir is not None else histogram_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if tag is None:
        stem = histogram_csv.stem
        for prefix in ("distributions_all_states_", "distributions_4pop_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        tag = stem

    df = pd.read_csv(histogram_csv)
    names = state_names_for_model(model_id)
    missing = [f"count_{name}" for name in names if f"count_{name}" not in df]
    if missing:
        raise ValueError("Missing nine-state counts in histogram CSV: " + ", ".join(missing))
    if "count_A" not in df:
        raise ValueError("Missing count_A in histogram CSV")
    n_all = int(df["count_A"].sum())
    if n_all <= 0:
        raise ValueError("No simulated samples in CSV")
    empirical_sum = df[[f"count_{name}" for name in names]].sum(axis=1)
    if not np.array_equal(empirical_sum, df["count_A"]):
        raise AssertionError("Nine populations do not partition A in every bin")
    edges = np.r_[df["left_x"].to_numpy(float), float(df["right_x"].iloc[-1])]
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("Non-increasing bin boundaries")
    dense = np.geomspace(0.001, min(2.0, max(1.76, float(edges[-1]))), 1000)
    theory = switching_mean_field_theory(model_id, switch_tsw, gamma, gamma_b, alpha)
    references = theoretical_density_and_bin_masses(theory, edges, dense, names=names)
    weights = np.asarray([theory["weights"][name] for name in names], dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise AssertionError("Nine theoretical states do not partition A")

    palette = {"sim": "#78DDE9", "outline": "#16899A", "theory": "#C06C31", "bins": "#36539D"}
    labels = [_latex_state_name(name) for name in names]
    figure_paths = {}
    rows = []
    binned = df[["left_x", "right_x", "center_x", "width_x",
                 "left_kb", "right_kb"]].copy()

    fig_cond, axes_cond = plt.subplots(3, 3, figsize=(17.0, 12.4), dpi=125)
    fig_global, axes_global = plt.subplots(3, 3, figsize=(17.0, 12.4), dpi=125)
    for ix, name in enumerate(names):
        counts = df[f"count_{name}"].to_numpy(dtype=np.int64)
        n = int(counts.sum())
        theory_weight = float(theory["weights"][name])
        sim_weight = n / n_all
        item = references[name]
        conditional_sim = counts / (n * widths) if n > 0 else np.full(len(widths), np.nan)
        global_sim = counts / (n_all * widths)
        conditional_th = item["bin_pdf"] if item is not None else np.full(len(widths), np.nan)
        global_th = conditional_th * theory_weight if item is not None else np.zeros(len(widths))
        smooth_cond_th = item["smooth"] if item is not None else None
        smooth_global_th = smooth_cond_th * theory_weight if item is not None else None
        masses_th = item["bin_masses"] if item is not None else np.zeros(len(widths))
        # State shape comparison and joint state+length comparison are different:
        # The former conditions on the state, the latter keeps the population weight.
        shape_tv = (float(0.5 * np.sum(np.abs(counts / n - masses_th)))
                    if n > 0 and item is not None else None)
        joint_l1 = float(np.sum(np.abs(counts / n_all - theory_weight * masses_th)))
        row = {
            "state": name, "blocking_code": SIM_STATE_CODES[name], "n": n,
            "fraction_sim": sim_weight, "fraction_theory": theory_weight,
            "difference_percentage_points": 100.0 * (sim_weight - theory_weight),
            "shape_TV_binned": shape_tv, "global_L1_binned": joint_l1,
            "mean_x_sim_binned": (float(np.dot(counts, df["center_x"]) / n)
                                  if n > 0 else None),
        }
        rows.append(row)
        binned[f"count_{name}"] = counts
        binned[f"sim_conditional_pdf_{name}_per_x"] = conditional_sim
        binned[f"theory_conditional_pdf_{name}_per_x"] = conditional_th
        binned[f"sim_global_pdf_{name}_per_x"] = global_sim
        binned[f"theory_global_pdf_{name}_per_x"] = global_th
        binned[f"theory_bin_mass_conditional_{name}"] = masses_th

        for mode, ax in (("conditional", axes_cond.flat[ix]),
                         ("global", axes_global.flat[ix])):
            if mode == "conditional":
                histogram = conditional_sim
                curve = smooth_cond_th
                theoretical_bin = conditional_th
            else:
                histogram = global_sim
                curve = smooth_global_th
                theoretical_bin = global_th
            if n > 0:
                ax.stairs(histogram, edges, baseline=0, fill=True,
                          facecolor=palette["sim"], edgecolor=palette["outline"],
                          alpha=0.58, linewidth=.55, label="Simulation")
            else:
                ax.text(.5, .5, "No sampled loops", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
            if curve is not None:
                ax.plot(dense, curve, color=palette["theory"], lw=1.8,
                        label="Theory")
                ax.stairs(theoretical_bin, edges, color=palette["bins"],
                          linewidth=1.15, linestyle="--", label="Theory (bins)")
            ax.set_xlim(.001, min(1.75, max(1.75, edges[-1])))
            ax.set_xscale("log")
            ax.set_ylim(bottom=0)
            ax.set_title(f"{labels[ix]}  |  sim {sim_weight:.3f}, th {theory_weight:.3f}",
                         fontsize=10.5)
            if ix % 3 == 0:
                ax.set_ylabel("Conditional PDF (1/x)" if mode == "conditional"
                              else "Global PDF (1/x)", fontsize=10)
            if ix // 3 == 2:
                ax.set_xlabel(r"Loop length $x=L/l_p$", fontsize=10)
            ax.grid(alpha=.16)
            if ix == 0 and curve is not None and n > 0:
                ax.legend(frameon=False, fontsize=8)

    subtitle = (f"{theory['kind']} | beta={theory['beta']:g}"
                if theory["beta"] is not None else theory["kind"])
    if model_id == 4:
        subtitle += " | Poisson proxy, NOT exact periodic theory"
    for mode, fig in (("conditional", fig_cond), ("global", fig_global)):
        fig.suptitle(f"Nine resolved collision states | {mode} PDF | {subtitle} | "
                     f"gamma={gamma:g}, gb={gamma_b:g}, alpha={alpha:g}",
                     fontsize=12, y=.995)
        fig.tight_layout(rect=(0, 0, 1, .974))
        path = output_dir / f"Plot_Theory_vs_Simulation_9states_{mode}_{tag}.png"
        fig.savefig(path, dpi=145, bbox_inches="tight")
        plt.close(fig)
        figure_paths[mode] = path

    # Per-state fractions and residuals directly expose probability transfer
    # between classes, which normalized PDFs alone cannot show.
    stats = pd.DataFrame(rows)
    xx = np.arange(len(names))
    fig, axs = plt.subplots(2, 1, figsize=(13.5, 7.5),
                            gridspec_kw={"height_ratios": [2.0, 1.1]}, dpi=135)
    axs[0].bar(xx-.2, stats["fraction_sim"], width=.4, color=palette["sim"],
               edgecolor=palette["outline"], lw=.6, label="Simulation")
    axs[0].bar(xx+.2, stats["fraction_theory"], width=.4, color=palette["theory"],
               alpha=.83, label="Theory")
    axs[0].set_ylabel("Share among all loop snapshots")
    axs[0].set_title(f"Population fractions: simulation versus theory | {subtitle}")
    axs[0].legend(frameon=False)
    axs[0].grid(axis="y", alpha=.18)
    delta = stats["difference_percentage_points"].to_numpy()
    axs[1].bar(xx, delta, color=np.where(delta >= 0, "#31827B", "#C07353"), width=.65)
    axs[1].axhline(0, lw=.9, color="black")
    axs[1].set_ylabel("Simulation - theory (pp)")
    axs[1].set_xticks(xx, labels)
    axs[1].grid(axis="y", alpha=.18)
    fig.tight_layout()
    p = output_dir / f"Plot_Theory_vs_Simulation_9states_fractions_{tag}.png"
    fig.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig)
    figure_paths["fractions"] = p

    # 3x3 state matrix: first leg is active (or left for model2), second
    # is passive (or right); rows/columns are free/b/c respectively.
    arr_sim = np.zeros((3, 3))
    arr_th = np.zeros((3, 3))
    state_code_to_i = {"f": 0, "b": 1, "c": 2}
    for row in rows:
        ix0, ix1 = (state_code_to_i[ch] for ch in row["blocking_code"])
        arr_sim[ix0, ix1] = row["fraction_sim"]
        arr_th[ix0, ix1] = row["fraction_theory"]
    matrix_delta = (arr_sim - arr_th) * 100.
    fig_m, axes_m = plt.subplots(1, 3, figsize=(13.2, 4.6), dpi=135)
    vmax = max(float(arr_sim.max()), float(arr_th.max()))
    for ax, mat, title, cmap, lo, hi, fmt in (
            (axes_m[0], arr_sim, "Simulation", "Blues", 0, vmax, ".3f"),
            (axes_m[1], arr_th, "Mean-field theory", "Blues", 0, vmax, ".3f"),
            (axes_m[2], matrix_delta, "Sim - theory (pp)", "RdBu_r",
             -max(1e-3, float(np.abs(matrix_delta).max())),
             max(1e-3, float(np.abs(matrix_delta).max())), "+.2f")):
        im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_xticks(range(3), ["free", "b", "c"])
        ax.set_yticks(range(3), ["free", "b", "c"])
        ax.set_xlabel("passive leg" if model_id != 2 else "right leg")
        ax.set_ylabel("active leg" if model_id != 2 else "left leg")
        ax.set_title(title)
        for ii in range(3):
            for jj in range(3):
                ax.text(jj, ii, format(mat[ii, jj], fmt),
                        ha="center", va="center", fontsize=11)
        fig_m.colorbar(im, ax=ax, fraction=.046, pad=.035)
    fig_m.suptitle("Nine state occupancies | " + subtitle, fontsize=12)
    fig_m.tight_layout(rect=(0, 0, 1, .94))
    p = output_dir / f"Plot_Theory_vs_Simulation_9states_matrix_{tag}.png"
    fig_m.savefig(p, dpi=145, bbox_inches="tight")
    plt.close(fig_m)
    figure_paths["matrix"] = p

    table_file = output_dir / f"Theory_vs_Simulation_9states_fractions_{tag}.csv"
    bin_file = output_dir / f"Theory_vs_Simulation_9states_bins_{tag}.csv"
    report_file = output_dir / f"Theory_vs_Simulation_9states_stats_{tag}.json"
    atomic_csv_df(table_file, stats)
    atomic_csv_df(bin_file, binned)
    report = {
        "theory_kind": theory["kind"], "model": int(model_id),
        "beta": (float(theory["beta"]) if theory["beta"] is not None else None),
        "gamma": float(gamma), "gamma_b": float(gamma_b), "alpha": float(alpha),
        "gamma_c_active": float(theory["gamma_c_active"]),
        "J": float(theory["J"]), "P": float(theory["P"]),
        "h": float(theory["h"]), "number_of_loop_snapshots": n_all,
        "simulation_state_fractions_sum": float(stats["fraction_sim"].sum()),
        "theory_state_fractions_sum": float(stats["fraction_theory"].sum()),
        "observations_are_temporally_correlated": True,
        "theory_for_model_4_is_poisson_surrogate": model_id == 4,
        "populations": rows,
        "figures": {key: str(value) for key, value in figure_paths.items()},
        "csv_fractions": str(table_file), "csv_bin_distributions": str(bin_file),
    }
    _write_json(report_file, report)
    print("Saved nine-state theory comparisons:")
    for path in figure_paths.values():
        print("  ", path)
    print("Saved nine-state diagnostics:", table_file)
    print("Largest population fraction residuals (percentage points):")
    for row in sorted(rows, key=lambda z: abs(z["difference_percentage_points"]),
                      reverse=True)[:5]:
        print("  %-12s sim=%0.4f theory=%0.4f delta=%+0.2f pp" % (
            row["state"], row["fraction_sim"], row["fraction_theory"],
            row["difference_percentage_points"]))
    return figure_paths, table_file, bin_file, report_file

def plot_four_distributions(csv_path, png_path):
    """Plot the four populations with global and conditional PDF normalizations."""
    df = pd.read_csv(csv_path)
    names = ("A", "A_bb", "A_b", "A_00")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=120)
    for ax, prefix, title in zip(
        axes, ("global_pdf", "conditional_pdf"),
        ("Global PDF: subgroup areas = fractions of A",
         "Conditional PDF: each subgroup area = 1"),
    ):
        for name in names:
            ax.plot(df["center_kb"], df[f"{prefix}_{name}_per_kb"],
                    label=name, linewidth=1.5)
        ax.set(xlabel="Loop length (kb)", ylabel="Density (1/kb)",
               title=title, xlim=(0, 800), yscale="log")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved four-population plot:", png_path)


def plot_nine_states(csv_path, png_path, names):
    """Plot nine raw and per-state-normalized densities."""
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)
    for ax, prefix, title in zip(
        axes, ("global_pdf", "conditional_pdf"),
        ("Nine states: common normalization by all loops",
         "Nine states: individually normalized PDFs")
    ):
        for name in names:
            ax.plot(df["center_kb"], df[f"{prefix}_{name}_per_kb"],
                    label=name, linewidth=1.2)
        ax.set(xlabel="Loop length (kb)", ylabel="Density (1/kb)",
               title=title, xlim=(0, 800), yscale="log")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved nine-state plot:", png_path)


def atomic_csv_df(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_csv_x(path):
    return pd.read_csv(path, usecols=["x"])["x"].to_numpy(dtype=np.float64)


def model1_lambda_x(gamma, gamma_b, alpha):
    """Expected mean full-loop length x=L/lp in model 1 (same as do_it)."""
    gamma_b_alf = gamma_b / (1.0 + alpha)
    B = gamma + 2.0 * (gamma_b_alf + 1.0)
    # Numerically stable positive root of gamma_dyn^2 + B*gamma_dyn = gamma.
    gamma_dyn = (2.0 * gamma) / (np.sqrt(B * B + 4.0 * gamma) + B)
    gamma_eff = 0.5 * (gamma + gamma_dyn) + gamma_b_alf
    return 1.0 / (1.0 + gamma_eff)


def model1_exponential_pdf_x(x, gamma, gamma_b, alpha):
    """Model-1 full-loop PDF. NOT a conditional theory for A_bb."""
    lam = model1_lambda_x(gamma, gamma_b, alpha)
    return np.exp(-np.asarray(x, dtype=float) / lam) / lam


def plot_distributions(a_samples, bb_samples, tag, path=None, show=False):
    """Two panels, globally normalized histogram and matched theoretical PDF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), dpi=130)
    x_min_plot = 0.001
    x_theory = np.geomspace(x_min_plot, .8, 1500)
    metrics = {}
    for ax, name, samples, color in zip(
            axes, ("A", "A_bb"), (a_samples, bb_samples),
            ("#1677b8", "#dc6b33")):
        edges, counts, density, nmae, nmae_shape = histogram_data(samples, name)
        if model != 4:
            # NMAE is measured against the model-4 reference in histogram_data.
            # It is not an appropriate goodness-of-fit metric for models 1, 2, or 3.
            nmae, nmae_shape = float("nan"), float("nan")
        metrics[f"nmae_{name}"] = nmae
        metrics[f"nmae_{name}_shape"] = nmae_shape
        ax.stairs(density, edges, fill=True, alpha=.28, color=color,
                  linewidth=1, label="Simulation (global PDF)")
        ax.stairs(density, edges, color=color, linewidth=.9)
        if model == 1:
            # The source only provides an exponential theory for *all* loops A.
            # The doubly barrier-contacting subset A_bb is conditional and is
            # NOT expected to obey the same exponential law.
            if name == "A":
                ax.plot(x_theory, model1_exponential_pdf_x(
                    x_theory, GAMMA_FOR_PLOT, GAMMA_B_FOR_PLOT, ALPHA_FOR_PLOT),
                    color="black", lw=1.75, ls=":",
                    label="Model 1 exponential theory (A)")
        ax.set(xscale="log", xlim=(x_min_plot, .8), xlabel="x = L/lp", ylabel="PDF in x")
        ax.set_xticks([0.001, 0.01, 0.1, 0.5])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "0.5"])
        if name == "A_bb" and model == 4:
            ax.set_title(f"{name}  (N={len(samples):,}; NMAE={nmae:.3f})", fontsize=11)
        else:
            # No valid model-4 NMAE reference for models 1, 2, or 3.
            ax.set_title(f"{name}  (N={len(samples):,})", fontsize=11)
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=9, loc="best")
    if model == 1:
        subtitle = ("\nModel 1: exponential reference is for A only; "
                    "there is no exponential A_bb theory in this script.")
    elif model == 4:
        subtitle = ("\nEmpirical plot only: the old fixed-gamma_c=3.7 theory is "
                    "not valid for arbitrary periodic switching rates.")
    elif model == 2:
        subtitle = ("\nModel 2: two-sided extrusion, both arms moving at v/2. "
                    "Model-4 reference theory is not used.")
    else:
        subtitle = ("\nModel 3: Poisson switching; shown simulation only."
                    " Theory model-4 reference is not used.")
    fig.suptitle(tag.replace("_", "  ") + subtitle,
                 fontsize=10.0, y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # PNG has already been generated before replacing destination.
        tmp = path.with_name(path.stem + ".partial.png")
        fig.savefig(tmp, dpi=130, bbox_inches="tight")
        os.replace(tmp, path)
    if show:
        plt.show()
    plt.close(fig)
    return metrics

def do_it(gamma, gamma_b, alpha, Plots=False, save_csv=True,
          return_stats=False, only_bb=False, show_progress=True,
          progress_interval=1.0, contact_width=0.01, save_dir=".",
          run_id=None, verbose=True):
    """Save A, A_bb, A_b and A_00 from recorded current b blockages."""
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive.")
    if model not in (0, 1, 2, 3, 4, 5):
        raise ValueError("Supported models: 0, 1, 2, 3, 4, 5.")
    if model in (3, 4, 5) and tsw <= 0:
        raise ValueError("tsw must be positive.")
    if model == 5 and not (0 <= tswMin < tsw):
        raise ValueError("Model 5 requires 0 <= tswMin < tsw.")
    if gamma <= 0 or gamma_b <= 0 or alpha <= 0:
        raise ValueError("This version, like the supplied source, needs gamma, gamma_b, alpha > 0.")

    gamma_b_alf = gamma_b / (1 + alpha)
    if model == 1:
        B = gamma + 2 * (gamma_b_alf + 1)
        gamma_dyn = (np.sqrt(B**2 + 4 * gamma) - B) / 2
        gamma_eff = (gamma + gamma_dyn) / 2 + gamma_b_alf
        lambda_t = 1 / (1 + gamma_eff)
    elif model == 2:
        B = gamma + 4 + 2 * gamma_b_alf
        gamma_dyn = (np.sqrt(B**2 + 16 * gamma) - B) / 2
        gamma_eff = (gamma + gamma_dyn) / 2 + gamma_b_alf
        lambda_t = 1 / (1 + gamma_eff / 2)
    else:
        lambda_t = None

    if verbose:
        print("Theoretical mean loop length (all states):", lambda_t if lambda_t is not None
              else "not calculated for this model")

    # Keep the exact RNG order of the original for generating all physical
    # objects. Fixed switching (mode 4) uses no further random draws.
    dist = tau * v / gamma
    scale = dist * n_ch
    t_start_c = np.random.uniform(low=0, high=t_full, size=n_all_c)
    t_life_c = np.random.exponential(scale=tau, size=n_all_c)
    pos_c = np.random.uniform(low=0, high=scale, size=n_all_c)
    if model in (1, 3, 4, 5):
        dir_c = np.random.randint(low=0, high=2, size=n_all_c)
    elif model == 0:
        dir_c = np.zeros(n_all_c)
    else:
        dir_c = 2 * np.ones(n_all_c)

    d_obst = v * tau / gamma_b
    n_b = round(scale / d_obst)
    n_b_initial = np.random.poisson(n_b)
    tau_b = tau / alpha
    t_start_initial = np.zeros(n_b_initial)
    t_life_initial = np.random.exponential(scale=tau_b, size=n_b_initial)
    pos_initial = np.random.uniform(0, scale, size=n_b_initial)
    n_births = np.random.poisson(n_b * t_full / tau_b)
    t_start_new = np.random.uniform(0, t_full, size=n_births)
    t_life_new = np.random.exponential(scale=tau_b, size=n_births)
    pos_new = np.random.uniform(0, scale, size=n_births)

    t_start_b = np.hstack((t_start_initial, t_start_new))
    t_life_b = np.hstack((t_life_initial, t_life_new))
    pos_b = np.hstack((pos_initial, pos_new))
    dir_b = (-1) * np.ones(n_b_initial + n_births)

    t_start = np.hstack((t_start_c, t_start_b))
    t_life = np.hstack((t_life_c, t_life_b))
    t_end = t_start + t_life
    pos = np.hstack((pos_c, pos_b))
    direction = np.hstack((dir_c, dir_b)).astype(np.int64)

    n_all = len(t_start)
    l_right = np.zeros(n_all, dtype=np.float64)
    l_left = np.zeros(n_all, dtype=np.float64)
    switch_counts = np.zeros(n_all_c, dtype=np.int64)
    # Store both b and c blocker object IDs. -1 means no confirmed blockage.
    left_blocker_id = np.full(n_all_c, -1, dtype=np.int64)
    right_blocker_id = np.full(n_all_c, -1, dtype=np.int64)
    if model in (3, 4, 5):
        t_next_sw = t_start_c + sample_switch_interval(n_all_c)
    else:
        t_next_sw = np.full(n_all_c, np.inf, dtype=np.float64)

    sw_times, sw_ids = _prepare_switch_events(model, t_start_c, t_end[:n_all_c],
                                               t_next_sw, t_full)

    start_order = np.argsort(t_start).astype(np.int64)
    end_order = np.argsort(t_end).astype(np.int64)
    active_ids = np.empty(n_all, dtype=np.int64)
    is_active = np.zeros(n_all, dtype=np.bool_)
    start_ptr = end_ptr = sw_ptr = n_active = n_total_cohesin_snapshots = 0
    n_b_collision_events = 0
    n_passive_c_events = 0
    dl = dt * v / 2 if model == 2 else dt * v

    # Numba dynamic lists avoid large worst-case buffer allocations and let
    # us retain EXACTLY the individual samples, without histogramming.
    a_out = List.empty_list(types.float64)
    abb_out = List.empty_list(types.float64)
    ab_out = List.empty_list(types.float64)
    a00_out = List.empty_list(types.float64)
    state_out = List()
    for _ in range(9):
        state_out.append(List.empty_list(types.float64))
    r_out = List.empty_list(types.float64)
    l_out = List.empty_list(types.float64)
    ids_out = List.empty_list(types.int64)

    time_grid = np.arange(0, t_full, dt)
    n_steps = len(time_grid)
    started = time.perf_counter()
    last_update = started
    if show_progress:
        print(f"Model {model}: {n_steps:,} steps, {n_all_c:,} loaded cohesins, "
              f"{n_all - n_all_c:,} total barrier events (JIT compilation on first chunk).")

    for first in range(0, n_steps, NUMBA_CHUNK_STEPS):
        last = min(n_steps, first + NUMBA_CHUNK_STEPS)
        (start_ptr, end_ptr, sw_ptr, n_active, n_total_cohesin_snapshots,
         n_b_collision_events, n_passive_c_events) = _simulate_chunk(
            time_grid[first:last], first, sample_every, t_full / 2,
            model, dc, dl, tsw, tau * v, PASSIVE_C_COLLISIONS,
            n_all_c, t_start, t_end, pos, direction,
            start_order, end_order, sw_times, sw_ids, t_next_sw,
            start_ptr, end_ptr, sw_ptr, active_ids, is_active, n_active,
            l_left, l_right, switch_counts, left_blocker_id, right_blocker_id,
            (save_csv or Plots), only_bb, a_out, abb_out, ab_out, a00_out, state_out,
            r_out, l_out, ids_out,
            n_total_cohesin_snapshots, n_b_collision_events, n_passive_c_events,
        )
        if show_progress:
            now = time.perf_counter()
            if last == n_steps or now - last_update >= progress_interval:
                frac = last / n_steps
                elapsed = now - started
                remaining = elapsed * (1 - frac) / max(frac, 1e-10)
                sys.stdout.write(f"\rModel {model}: {frac:6.1%} ({last:,}/{n_steps:,}) "
                                 f"elapsed {format_duration(elapsed)} "
                                 f"ETA {format_duration(remaining)}  ")
                sys.stdout.flush()
                last_update = now
    if show_progress:
        print()

    # Convert individual snapshots directly, no intermediate list of empty
    # arrays and no second filtering pass.
    a_samples = np.asarray(a_out, dtype=np.float64)
    bb_samples = np.asarray(abb_out, dtype=np.float64)
    ab_samples = np.asarray(ab_out, dtype=np.float64)
    a00_samples = np.asarray(a00_out, dtype=np.float64)
    if len(a_samples) != len(bb_samples) + len(ab_samples) + len(a00_samples):
        raise AssertionError("A != A_bb + A_b + A_00: populations overlap or are missing")
    state_names = state_names_for_model(model)
    state_arrays = {name: np.asarray(state_out[k], dtype=np.float64)
                    for k, name in enumerate(state_names)}
    if sum(len(v) for v in state_arrays.values()) != len(a_samples):
        raise AssertionError("A != sum of nine collision states")
    # Check conservation in coarse aggregates separately.
    if model == 2:
        bb_name = "A_Lb_Rb"
        single_b = ("A_L0_Rb", "A_Lc_Rb", "A_Lb_R0", "A_Lb_Rc")
        no_b = ("A_L0_R0", "A_L0_Rc", "A_Lc_R0", "A_Lc_Rc")
    else:
        bb_name = "Abb"
        single_b = ("Ab1", "Abp", "Abc", "Acb")
        no_b = ("A11", "Ac1", "Acp", "Acc")
    assert len(bb_samples) == len(state_arrays[bb_name])
    assert len(ab_samples) == sum(len(state_arrays[k]) for k in single_b)
    assert len(a00_samples) == sum(len(state_arrays[k]) for k in no_b)
    length_right = np.asarray(r_out, dtype=np.float64)
    length_left = np.asarray(l_out, dtype=np.float64)
    sampled_ids = np.asarray(ids_out, dtype=np.int64)
    length_hist = length_right + length_left

    mean_loop = float(np.mean(length_hist)) if length_hist.size else float("nan")
    rho = (float(np.corrcoef(length_right, length_left)[0, 1])
           if length_hist.size > 1 and np.std(length_right) > 0 and np.std(length_left) > 0
           else float("nan"))

    tag = number_tag(gamma, gamma_b, alpha, contact_width, run_id=run_id)
    output = Path(save_dir).expanduser()
    a_file = output / f"A_{tag}.csv"
    bb_file = output / f"A_bb_{tag}.csv"
    ab_file = output / f"A_b_{tag}.csv"
    a00_file = output / f"A_00_{tag}.csv"
    plot_file = output / f"Plot_{tag}.png"
    if save_csv:
        atomic_csv_x(a_file, a_samples)
        atomic_csv_x(bb_file, bb_samples)
        atomic_csv_x(ab_file, ab_samples)
        atomic_csv_x(a00_file, a00_samples)
        for name, sample in state_arrays.items():
            atomic_csv_x(output / f"{name}_{tag}.csv", sample)
        _write_json(output / f"classification_{tag}.json", {
            "classification": "current confirmed b/c blockages with surviving obstacles",
            "nine_state_convention": ("left/right; both arms active" if model == 2
                                      else "active/passive; exactly one active arm"),
            "nine_state_names": list(state_names),
            "nine_state_counts": {name: int(len(v)) for name, v in state_arrays.items()},
            "nine_state_fractions": {name: (len(v)/len(a_samples) if len(a_samples) else None)
                                      for name, v in state_arrays.items()},
            "states_with_c": "record collisions with other cohesins (type c)",
            "passive_c_collisions_enabled": bool(PASSIVE_C_COLLISIONS),
            "passive_c_collision_events": int(n_passive_c_events),
            "passive_c_rule": ("reciprocal event: moving c tip contacts free passive "
                               "c tip and blocks on both sides; b blockers win"),
            "definition": {
                "A": "all sampled loops",
                "A_bb": "both legs currently blocked by distinct b obstacles",
                "A_b": "at least one leg currently blocked by b, except A_bb",
                "A_00": "no leg currently blocked by b",
            },
            "collision_geometry_dc": float(dc),
            "size_unused_for_classification": float(contact_width),
            "confirmed_b_collision_events": int(n_b_collision_events),
            "counts": {
                "A": int(len(a_samples)), "A_bb": int(len(bb_samples)),
                "A_b": int(len(ab_samples)), "A_00": int(len(a00_samples)),
            },
            "fractions": {
                "A_bb": len(bb_samples) / len(a_samples) if len(a_samples) else None,
                "A_b": len(ab_samples) / len(a_samples) if len(a_samples) else None,
                "A_00": len(a00_samples) / len(a_samples) if len(a_samples) else None,
            },
        })
        all_arrays = {"A": a_samples, "A_bb": bb_samples,
                      "A_b": ab_samples, "A_00": a00_samples, **state_arrays}
        binned_file = export_population_histograms(output, tag, all_arrays)
        # A convenient second filename with the same information, emphasizing
        # the fact that the table now contains all nine individual states.
        nine_file = output / f"distributions_all_states_{tag}.csv"
        atomic_csv_df(nine_file, pd.read_csv(binned_file))
        if Plots:
            plot_four_distributions(
                binned_file, output / f"Plot_4pop_{tag}.png")
            plot_nine_states(
                binned_file, output / f"Plot_9states_{tag}.png", state_names)
            if THEORY_COMPARISON_PLOTS:
                plot_theory_vs_simulation_four_states(
                    model, tsw, gamma, gamma_b, alpha, state_arrays,
                    a_samples, binned_file, output, tag)
                plot_theory_vs_simulation_nine_states(
                    model, tsw, gamma, gamma_b, alpha,
                    binned_file, output, tag)
        if verbose:
            print("Saved raw A samples:", a_file)
            print("Saved raw A_bb samples:", bb_file)
            print("Saved raw A_b samples:", ab_file)
            print("Saved raw A_00 samples:", a00_file)
            print("Nine collision states:",
                  {name: len(vals) for name, vals in state_arrays.items()})
            print(f"Population sizes: N_A={len(a_samples)}, "
                  f"N_bb={len(bb_samples)}, N_b={len(ab_samples)}, "
                  f"N_00={len(a00_samples)}")
    if verbose:
        print("Confirmed b-collision events:", int(n_b_collision_events))
        print("New passive-arm c-collision events:", int(n_passive_c_events))
        print("gamma, substrate length, mean A:",
              gamma, scale, float(np.mean(a_samples)) if len(a_samples) else float("nan"))
        if only_bb:
            fraction = (len(length_hist) / n_total_cohesin_snapshots
                        if n_total_cohesin_snapshots else float("nan"))
            print(f"A_bb snapshots: {len(length_hist)} / {n_total_cohesin_snapshots}"
                  f" ({fraction:.2%}); mean and rho refer to A_bb only")
        print("Arm correlation rho:", rho)
        if model in (3, 4, 5):
            print("Total direction switches:", int(switch_counts.sum()))
    # Plot A and A_bb from the SAME run, regardless of only_bb.
    if Plots:
        plot_distributions(a_samples, bb_samples, tag,
                           path=plot_file if save_csv else None, show=False)

    if return_stats:
        return {
            "model": model,
            "mean_loop": mean_loop,
            "mean_A": float(np.mean(a_samples)) if len(a_samples) else float("nan"),
            "rho": rho,
            "switches": int(switch_counts.sum()),
            "confirmed_b_collision_events": int(n_b_collision_events),
            "passive_c_collision_events": int(n_passive_c_events),
            "passive_c_collisions_enabled": bool(PASSIVE_C_COLLISIONS),
            "n_snapshots": len(length_hist),
            "n_total_cohesin_snapshots": n_total_cohesin_snapshots,
            "fraction_bb": (len(length_hist) / n_total_cohesin_snapshots
                            if only_bb and n_total_cohesin_snapshots else None),
            "only_bb": only_bb,
            "theory_mean": lambda_t,
            "tag": tag,
            "file_A": str(a_file) if save_csv else None,
            "file_A_bb": str(bb_file) if save_csv else None,
            "file_A_b": str(ab_file) if save_csv else None,
            "file_A_00": str(a00_file) if save_csv else None,
            "n_A": len(a_samples),
            "n_A_bb": len(bb_samples),
            "n_A_b": len(ab_samples),
            "n_A_00": len(a00_samples),
            "states9_counts": {name: len(v) for name, v in state_arrays.items()},
            "nmae_A_bb": histogram_data(bb_samples, "A_bb")[3]
                          if model == 4 and len(bb_samples) else float("nan"),
        }
    return [mean_loop]


def _grid_values(text):
    vals = [float(x) for x in text.split(",") if x.strip()]
    if not vals or any(not np.isfinite(x) or x <= 0 for x in vals):
        raise ValueError("Grid values must be positive finite decimal numbers")
    if len(set(vals)) != len(vals):
        raise ValueError("Grid values must not be repeated")
    return vals


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _row_from_files(dc_value, size_value, tsw_value, rep, seed, fa, fb,
                    fc, fd, multiplier_value=None):
    a = load_csv_x(fa)
    b = load_csv_x(fb)
    c = load_csv_x(fc)
    d = load_csv_x(fd)
    assert len(a) == len(b) + len(c) + len(d)
    nmae_b = (histogram_data(b, "A_bb")[3]
              if model == 4 and len(b) else float("nan"))
    return {
        "dc": dc_value, "size": size_value, "tsw": tsw_value,
        "multiplier": multiplier_value if multiplier_value is not None else np.nan,
        "repeat": rep, "seed": seed, "n_A": len(a), "n_A_bb": len(b),
        "n_A_b": len(c), "n_A_00": len(d),
        "mean_A": float(np.mean(a)) if len(a) else float("nan"),
        "mean_A_bb": float(np.mean(b)) if len(b) else float("nan"),
        "CDF_A_bb_002": float(np.mean(b < .02)) if len(b) else float("nan"),
        "nmae_A_bb": nmae_b,
        "file_A": str(fa), "file_A_bb": str(fb),
        "file_A_b": str(fc), "file_A_00": str(fd),
    }


def run_grid(args):
    global model, dc, tsw, n_all_c, n_ch, t_full, dt, sample_dt, sample_every
    global GAMMA_FOR_PLOT, GAMMA_B_FOR_PLOT, ALPHA_FOR_PLOT, PASSIVE_C_COLLISIONS
    global THEORY_COMPARISON_PLOTS
    PASSIVE_C_COLLISIONS = args.passive_c_collisions
    THEORY_COMPARISON_PLOTS = not args.no_theory_plots
    model = args.model
    n_all_c = args.n_all_c if args.n_all_c is not None else 10000
    n_ch = args.n_ch
    dt = args.dt
    sample_dt = args.sample_dt
    sample_every = max(1, round(sample_dt / dt))
    t_full = round(n_all_c * tau / n_ch)

    dc_grid = _grid_values(args.dc_grid)
    multiplier_grid = (_grid_values(args.multiplier_grid)
                       if args.multiplier_grid is not None else
                       ([args.multiplier] if args.multiplier is not None else None))
    # In ratio mode, each column dc has its own size values: s = m * dc.
    # Otherwise preserve the original Cartesian dc x size grid.
    size_grid = (_grid_values(args.size_grid or ".015")
                 if multiplier_grid is None else None)
    n_sizes_per_dc = (len(multiplier_grid) if multiplier_grid is not None
                      else len(size_grid))
    if model in (1, 2):
        # Models 1 and 2 never switch directions; avoid redundant simulations.
        # Store tsw=0 in filenames to make this explicit.
        tsw_grid = [0.0]
        print(f"Model {model}: --tsw-grid is not applicable; "
              "one run per (dc, size, repeat).")
    else:
        tsw_grid = _grid_values(args.tsw_grid)
    gamma = 2.7 if args.gamma is None else args.gamma
    GAMMA_FOR_PLOT, GAMMA_B_FOR_PLOT, ALPHA_FOR_PLOT = gamma, args.gamma_b, args.alpha
    base = Path(args.save_dir).expanduser()
    grid_name = (f"grid_collision_9states_pc{int(PASSIVE_C_COLLISIONS)}_{STATE_CLASSIFICATION_VERSION}_m{model}_n{n_all_c}_nch{n_ch}_g{gamma:g}"
                 f"_gb{args.gamma_b:g}_alpha{args.alpha:g}"
                 f"_dt{dt:g}_r{args.repeats}_seed{args.seed}")
    if multiplier_grid is not None:
        # Separate output folders from old fixed-size grids, avoiding cache
        # collisions and preserving old data, even for the same random seed.
        m_tag = "-".join(f"{m:g}" for m in multiplier_grid)
        if len(m_tag) > 75:
            import hashlib
            digest = hashlib.sha256(m_tag.encode("ascii")).hexdigest()[:12]
            m_tag = f"{len(multiplier_grid)}v_{digest}"
        grid_name += (f"_mul{m_tag}" if len(multiplier_grid) == 1
                      else f"_mgrid{m_tag}")
    root = base / grid_name
    raw_dir, plot_dir = root / "raw", root / "plots"
    settings = dict(model=model, n_all_c=n_all_c, n_ch=n_ch,
                    gamma=gamma, gamma_b=args.gamma_b, alpha=args.alpha,
                    dt=dt, sample_dt=sample_dt, tau=tau, v=v,
                    passive_c_collisions=bool(PASSIVE_C_COLLISIONS),
                    repeats=args.repeats, seed=args.seed,
                    dc_grid=dc_grid, size_grid=size_grid, tsw_grid=tsw_grid,
                    lp_mb=LP_MB, classification="recorded active b-blocker IDs, not proximity",
                    theory_gamma_c=THEORY_GAMMA_C,
                    theory_gamma_b=THEORY_GAMMA_B,
                    theory_alpha=THEORY_ALPHA,
                    theoretical_areas_in_xi=THEORY_AREAS_XI,
                    note=("Model 1: no direction switching (tsw=0 is a filename tag only). "
                          "The exponential theory is for the unconditional A distribution, "
                          "not the conditional A_bb. Any gamma_c=3.7 theory metadata "
                          "are retained only for comparison with the model-4 reference."
                          if model == 1 else
                          "Simulation gamma=2.7 is BARE cohesin density; theory "
                          "gamma_c=3.7 INCLUDES the dynamic correction. "
                          "Fixed analytical formula; PDF globally normalized in x. "
                          "size is ignored for event-based classification."))
    if multiplier_grid is not None:
        settings["size_rule"] = "size = multiplier * dc"
        settings["multiplier_grid"] = multiplier_grid
        settings["size_unit"] = "dimensionless (same as dc)"
    config_path = root / "grid_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != settings:
            # Allow an earlier model-1 metadata-only note, without changing
            # numerical parameters, CSV names, seeds or resumption behavior.
            comparable_old = {k: v for k, v in old.items() if k != "note"}
            comparable_new = {k: v for k, v in settings.items() if k != "note"}
            if model != 1 or comparable_old != comparable_new:
                raise ValueError("Existing grid_config.json differs. Choose another output folder.")
    else:
        _write_json(config_path, settings)
    run_table = root / "metrics_by_run.csv"
    config_table = root / "metrics_by_config.csv"
    runs = {}
    summary = {}
    if run_table.exists():
        for row in pd.read_csv(run_table).to_dict("records"):
            runs[(float(row["dc"]), float(row["size"]),
                  float(row["tsw"]), int(row["repeat"]))] = row
    if config_table.exists():
        for row in pd.read_csv(config_table).to_dict("records"):
            summary[(float(row["dc"]), float(row["size"]), float(row["tsw"]))] = row
    total = len(dc_grid) * n_sizes_per_dc * len(tsw_grid) * args.repeats
    second_axis = "multipliers" if multiplier_grid is not None else "sizes"
    print(f"Model {model}: {len(dc_grid)} dc x {n_sizes_per_dc} {second_axis} "
          f"x {len(tsw_grid)} tsw x {args.repeats} repeats = {total} runs. "
          f"Saving to: {root}", flush=True)
    print("Classification is based on recorded b collision; size is metadata only.")
    if multiplier_grid is not None:
        print("LEGACY WIDTH RULE: size = multiplier * dc (no effect on classes)")
        for dc_value in dc_grid:
            print(f"  dc={dc_value:g} -> " + ", ".join(
                f"m={m:g}: size={m * dc_value:g}" for m in multiplier_grid))
    if (model == 4 and (abs(gamma - 2.7) > 1e-12 or
        abs(args.gamma_b - THEORY_GAMMA_B) > 1e-12 or
        abs(args.alpha - THEORY_ALPHA) > 1e-12)):
        print("NOTE: fixed reference theory is for dynamic-corrected gamma_c=3.7, "
              "gamma_b=4, alpha=1.2. Only the SIMULATION parameters were changed.")
    if args.dry_run:
        return
    new_runs = 0
    stopped = False
    for di, dc_value in enumerate(dc_grid):
        dc = dc_value
        size_settings = ([(m, m * dc_value) for m in multiplier_grid]
                         if multiplier_grid is not None else
                         [(None, width) for width in size_grid])
        for si, (multiplier_value, size_value) in enumerate(size_settings):
            for ti, tsw_value in enumerate(tsw_grid):
                tsw = tsw_value
                file_pairs = []
                all_done = True
                for rep in range(1, args.repeats + 1):
                    seed = args.seed + 100000 * di + 1000 * si + 30 * ti + rep
                    run_name = f"rep{rep:02d}_seed{seed}"
                    tag = number_tag(gamma, args.gamma_b, args.alpha,
                                     size_value, run_id=run_name)
                    fa = raw_dir / f"A_{tag}.csv"
                    fb = raw_dir / f"A_bb_{tag}.csv"
                    fc = raw_dir / f"A_b_{tag}.csv"
                    fd = raw_dir / f"A_00_{tag}.csv"
                    nine_files = tuple(raw_dir / f"{name}_{tag}.csv"
                                       for name in state_names_for_model(model))
                    if not (fa.is_file() and fb.is_file() and
                            fc.is_file() and fd.is_file() and
                            all(p.is_file() for p in nine_files)):
                        if args.max_new_runs and new_runs >= args.max_new_runs:
                            stopped = True
                            all_done = False
                            break
                        np.random.seed(seed)
                        do_it(gamma, args.gamma_b, args.alpha,
                              Plots=False, save_csv=True, return_stats=False,
                              show_progress=False, only_bb=False,
                              contact_width=size_value, save_dir=raw_dir,
                              run_id=run_name, verbose=False)
                        new_runs += 1
                        ratio_info = (f", m={multiplier_value:g}"
                                      if multiplier_value is not None else "")
                        print(f"SAVE {new_runs} / {total}: "
                              f"dc={dc_value:g}, size={size_value:g}{ratio_info}, "
                              f"tsw={tsw_value:g}, rep={rep}", flush=True)
                    else:
                        if args.verbose_skips:
                            print("SKIP", tag)
                    file_pairs.append((fa, fb, fc, fd, *nine_files))
                    key = (dc_value, size_value, tsw_value, rep)
                    if key not in runs or pd.isna(runs[key].get("n_A_bb")):
                        runs[key] = _row_from_files(
                            dc_value, size_value, tsw_value, rep, seed, fa, fb,
                            fc, fd, multiplier_value=multiplier_value)
                        atomic_csv_df(run_table, pd.DataFrame(runs.values()))
                if all_done and len(file_pairs) == args.repeats:
                    config_tag = number_tag(gamma, args.gamma_b, args.alpha,
                                            size_value, run_id="pooled")
                    plot_file = plot_dir / f"Plot_{config_tag}.png"
                    key = (dc_value, size_value, tsw_value)
                    comparison_file = (plot_dir / f"Plot_Theory_vs_Simulation_4states_{config_tag}.png")
                    nine_comparison_file = (plot_dir / f"Plot_Theory_vs_Simulation_9states_conditional_{config_tag}.png")
                    if (key not in summary or not plot_file.is_file()
                            or (THEORY_COMPARISON_PLOTS and
                                (not comparison_file.is_file() or not nine_comparison_file.is_file()))):
                        a = np.concatenate([load_csv_x(v[0]) for v in file_pairs])
                        b = np.concatenate([load_csv_x(v[1]) for v in file_pairs])
                        c = np.concatenate([load_csv_x(v[2]) for v in file_pairs])
                        d = np.concatenate([load_csv_x(v[3]) for v in file_pairs])
                        assert len(a) == len(b) + len(c) + len(d)
                        nine_names = state_names_for_model(model)
                        nine_arrays = {
                            name: np.concatenate([load_csv_x(v[4+i]) for v in file_pairs])
                            for i, name in enumerate(nine_names)
                        }
                        combined_csv = export_population_histograms(plot_dir, config_tag, {
                            "A": a, "A_bb": b, "A_b": c, "A_00": d, **nine_arrays
                        })
                        atomic_csv_df(plot_dir / f"distributions_all_states_{config_tag}.csv",
                                      pd.read_csv(combined_csv))
                        plot_nine_states(combined_csv,
                                         plot_dir / f"Plot_9states_{config_tag}.png",
                                         nine_names)
                        plot_four_distributions(
                            combined_csv, plot_dir / f"Plot_4pop_{config_tag}.png")
                        if THEORY_COMPARISON_PLOTS:
                            plot_theory_vs_simulation_four_states(
                                model, tsw_value, gamma, args.gamma_b, args.alpha,
                                nine_arrays, a, combined_csv, plot_dir, config_tag)
                            plot_theory_vs_simulation_nine_states(
                                model, tsw_value, gamma, args.gamma_b, args.alpha,
                                combined_csv, plot_dir, config_tag)
                        metrics = plot_distributions(a, b, config_tag, plot_file, False)
                        summary[key] = dict(dc=dc_value, size=size_value,
                                            multiplier=(multiplier_value if multiplier_value is not None else np.nan),
                                            tsw=tsw_value, n_A=len(a), n_A_bb=len(b),
                                            n_A_b=len(c), n_A_00=len(d),
                                            mean_A=float(np.mean(a)) if len(a) else float("nan"),
                                            mean_A_bb=float(np.mean(b)) if len(b) else float("nan"),
                                            CDF_A_bb_002=float(np.mean(b < .02)) if len(b) else float("nan"),
                                            nmae_A_bb=metrics["nmae_A_bb"],
                                            nmae_A_bb_shape=metrics["nmae_A_bb_shape"],
                                            plot_file=str(plot_file))
                        atomic_csv_df(config_table, pd.DataFrame(summary.values()))
                if stopped:
                    break
            if stopped:
                break
        if stopped:
            break
    if summary and model == 4:
        ranked = pd.DataFrame(summary.values())
        ranked = ranked[ranked.n_A_bb >= args.min_bb_samples]
        ranked = ranked.sort_values("nmae_A_bb", na_position="last")
        atomic_csv_df(root / "ranking_by_NMAE.csv", ranked)
    print(f"New runs: {new_runs}; complete parameter points: {len(summary)}; "
          f"total combinations: {total // args.repeats}. Output: {root}", flush=True)


def main(argv=None):
    global model, dc, tsw, n_all_c, n_ch, t_full, dt, sample_dt, sample_every
    global GAMMA_FOR_PLOT, GAMMA_B_FOR_PLOT, ALPHA_FOR_PLOT, PASSIVE_C_COLLISIONS
    global THEORY_COMPARISON_PLOTS
    p = argparse.ArgumentParser(
        description="Cohesin Numba simulation: 4 aggregate populations + 9 b/c collision states")
    p.add_argument("--grid", action="store_true", help="Run parameter grid for models 1/2/3/4")
    p.add_argument("--model", type=int, choices=(1, 2, 3, 4), default=3,
                   help="1 = one-sided fixed; 2 = two-sided; "
                        "3 = Poisson switching; 4 = periodic switching")
    p.add_argument("--save-dir", default="/content/drive/MyDrive/Cohesin_trials"
                   if Path("/content/drive/MyDrive").exists() else ".")
    p.add_argument("--n-all-c", type=int, default=None,
                   help="Default: 10000 for grid, 1000000 for single run")
    p.add_argument("--n-ch", type=int, default=50)
    p.add_argument("--dt", type=float, default=.001)
    p.add_argument("--sample-dt", type=float, default=.1)
    p.add_argument("--gamma", type=float, default=None,
                   help="BARE cohesin density, default 2.7 (single AND grid). "
                        "Reference theoretical gamma_c=3.7 already includes the dynamic correction.")
    p.add_argument("--gamma-b", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=1.2)
    p.add_argument("--dc", type=float, default=.015, help="Collision tolerance/geometry dc")
    p.add_argument("--size", type=float, default=None,
                   help="Single-run observational contact width (default .015); "
                        "alternative to --multiplier")
    ratios = p.add_mutually_exclusive_group()
    ratios.add_argument("--multiplier", type=float, default=None,
                        help="Set size = multiplier * dc; works in single and grid modes")
    ratios.add_argument("--multiplier-grid", default=None,
                        help="For --grid: comma-separated multipliers; sweep size=m*dc")
    p.add_argument("--tsw", type=float, default=.01, help="Single-run switch interval")
    p.add_argument("--dc-grid", default=".001,.004,.007,.010,.013,.016,.019,.020")
    p.add_argument("--size-grid", default=None,
                   help="Legacy widths used only for metadata; collision classes depend on dc")
    p.add_argument("--tsw-grid", default=".005,.008,.011,.014,.017,.020")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--max-new-runs", type=int, default=0,
                   help="Limit completed new grid runs; 0 means unlimited")
    p.add_argument("--min-bb-samples", type=int, default=100,
                   help="Require at least this many A_bb samples for ranked table")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose-skips", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="Do not plot single run")
    p.add_argument("--no-theory-plots", action="store_true",
                   help="Disable all theory-vs-simulation plots, 4-state and 9-state")
    p.add_argument("--compare-existing", default=None,
                   help="Rebuild nine-state theory plots directly from an existing "
                        "distributions_all_states_*.csv; NO simulation is run")
    p.add_argument("--compare-output", default=None,
                   help="Output directory for --compare-existing; default next to CSV")
    p.add_argument("--passive-c-collisions", action=argparse.BooleanOptionalAction,
                   default=True, help="Register c-collision on passive neighbor tip "
                   "when the other cohesin's moving tip hits it (default: enabled). "
                   "Use --no-passive-c-collisions for control runs.")
    args = p.parse_args(argv)
    if args.n_ch < 1 or args.dt <= 0 or args.sample_dt <= 0 or args.repeats < 1:
        p.error("n_ch, dt, sample_dt and repeats must be positive")
    if args.multiplier is not None and (not np.isfinite(args.multiplier)
                                        or args.multiplier <= 0):
        p.error("--multiplier must be a positive finite number")
    if args.size is not None and (not np.isfinite(args.size) or args.size <= 0):
        p.error("--size must be a positive finite number")
    ratio_mode = args.multiplier is not None or args.multiplier_grid is not None
    if ratio_mode and (args.size is not None or args.size_grid is not None):
        p.error("Use --multiplier/--multiplier-grid OR --size/--size-grid, not both")
    if args.multiplier_grid is not None and not args.grid:
        p.error("--multiplier-grid requires --grid; use --multiplier for single runs")
    if args.compare_existing is not None:
        comparison_tsw = 0.0 if args.model == 2 else args.tsw
        plot_theory_vs_simulation_nine_states(
            args.model, comparison_tsw,
            (2.7 if args.gamma is None else args.gamma),
            args.gamma_b, args.alpha, args.compare_existing,
            args.compare_output)
        return
    if args.grid:
        run_grid(args)
        return
    model = args.model
    PASSIVE_C_COLLISIONS = args.passive_c_collisions
    THEORY_COMPARISON_PLOTS = not args.no_theory_plots
    n_all_c = args.n_all_c if args.n_all_c is not None else 1000000
    n_ch = args.n_ch
    dc = args.dc
    tsw = 0.0 if model == 2 else args.tsw
    if model == 2:
        print("Model 2: symmetric two-sided extrusion; "
              "--tsw is ignored and both legs move at v/2.")
    dt = args.dt
    sample_dt = args.sample_dt
    sample_every = max(1, round(sample_dt / dt))
    t_full = round(n_all_c * tau / n_ch)
    gamma = 2.7 if args.gamma is None else args.gamma
    GAMMA_FOR_PLOT, GAMMA_B_FOR_PLOT, ALPHA_FOR_PLOT = gamma, args.gamma_b, args.alpha
    width = dc * args.multiplier if args.multiplier is not None else (
        args.size if args.size is not None else .015)
    print("Classification: confirmed b AND c collision events, 9 states and 4 aggregates;")
    print("--size affects filenames only, while --dc determines collision geometry.")
    print("Reciprocal passive-c collision:", "ON" if PASSIVE_C_COLLISIONS else "OFF")
    if args.multiplier is not None:
        print(f"Single run: dc={dc:g}, multiplier={args.multiplier:g}, "
              f"legacy size={width:g} (ignored for classes)")
    np.random.seed(args.seed)
    do_it(gamma, args.gamma_b, args.alpha, Plots=not args.no_plots,
          only_bb=(model != 2), contact_width=width, save_csv=True,
          save_dir=args.save_dir)


if __name__ == "__main__":
    main()
