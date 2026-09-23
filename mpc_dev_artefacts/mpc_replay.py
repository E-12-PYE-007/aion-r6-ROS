#!/usr/bin/env python3
"""Headless closed-loop replay of UnicycleMPC against the real esdf_single_obs
bag, mirroring what mpc_path_follower.py + chunk_generator.py do, without the
ROS graph or any real-time constraint. Used throughout the MPC-obstacle-avoid
investigation (see MPC_FINDINGS.md at the repo root) to sweep parameters fast -
each run re-imports the solver fresh, so constants.py edits always take effect.

Requires the ROS environment sourced (for rclpy/nvblox_msgs deserialization) and
the control/debug packages built - `source install/setup.bash` from the repo
root, then run this directly.

Prints a one-line summary; pass --verbose for the tick-by-tick trace.

Examples:
    python3 mpc_dev_artefacts/mpc_replay.py                              # current constants.py, as committed
    python3 mpc_dev_artefacts/mpc_replay.py --horizon-n 14 --low-v-weight 0  # original deadlock, no shaping
    python3 mpc_dev_artefacts/mpc_replay.py --qf-multiplier 100 --low-v-weight 0  # early Qf-cranking experiment
    python3 mpc_dev_artefacts/mpc_replay.py --low-v-weight 100 --low-v-threshold 0.15  # attempt #12
    python3 mpc_dev_artefacts/mpc_replay.py --verbose --ticks 90          # tick-by-tick trace
"""
import sys
import sqlite3
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src' / 'control'))
sys.path.insert(0, str(REPO / 'src' / 'debug'))

from rclpy.serialization import deserialize_message
from nvblox_msgs.msg import DistanceMapSlice
from control.mpc_imp.esdf_map import EsdfMap
from control.mpc_imp.solver_setup import UnicycleMPC
from control.mpc_imp.constants import CONTROL_LOOP_DT
from debug.mpc_imp_test.straight_path import StraightPath, PATH_ORIGIN, PATH_HEADING

BAG = REPO / 'mpc_dev_artefacts' / 'rosbags' / 'esdf_single_obs' / 'esdf_single_obs_0.db3'
LOOKAHEAD_M = 0.8
N_WAYPOINTS = 8
WAYPOINT_SPACING_M = 0.1
WAYPOINT_DT = 1.0 / 3.0


def load_esdf():
    con = sqlite3.connect(str(BAG))
    cur = con.cursor()
    topic_id = [t[0] for t in cur.execute('SELECT id, name, type FROM topics').fetchall()
                if t[1] == '/nvblox_node/static_map_slice'][0]
    row = cur.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1',
                       (topic_id,)).fetchone()
    msg = deserialize_message(row[0], DistanceMapSlice)
    esdf = EsdfMap()
    esdf.store(msg)
    return esdf


def run(n_ticks, patch_resolution, verbose, qf_multiplier=None, horizon_n=None,
        low_v_weight=None, low_v_threshold=None, path_origin=None, path_heading=None):
    esdf = load_esdf()
    kwargs = dict(patch_resolution=patch_resolution, safety_margin=0.20)
    if qf_multiplier is not None:
        from control.mpc_imp.constants import QF
        kwargs['Q_f'] = np.asarray(QF) * qf_multiplier
    if horizon_n is not None:
        kwargs['N'] = horizon_n
        kwargs['M'] = horizon_n
    if low_v_weight is not None:
        kwargs['low_v_weight'] = low_v_weight
    if low_v_threshold is not None:
        kwargs['low_v_threshold'] = low_v_threshold
    mpc = UnicycleMPC(**kwargs)
    origin = PATH_ORIGIN if path_origin is None else tuple(path_origin)
    heading = PATH_HEADING if path_heading is None else path_heading
    path_line = StraightPath(origin, heading)

    # Robot starts facing the line's own heading, not just theta=0 - matters once
    # the line isn't axis-aligned, else the first few ticks are spent correcting a
    # large initial heading mismatch rather than reacting to the obstacle itself.
    x = np.array([origin[0], origin[1], heading])
    mpc.reset()
    psi = 0.0

    max_slack_seen = 0.0
    n_fail = 0
    xs = []

    for tick in range(n_ticks):
        # Reproduces chunk_generator.py's own logic: aim LOOKAHEAD_M ahead on the
        # (obstacle-blind) reference line, lay out N_WAYPOINTS evenly along the
        # segment from the current pose to that point.
        target_xy = path_line.lookahead_point(x[:2], LOOKAHEAD_M)
        seg = target_xy - x[:2]
        seg_dist = np.linalg.norm(seg)
        seg_dir = seg / seg_dist if seg_dist > 1e-9 else np.array([1.0, 0.0])
        heading = np.arctan2(seg_dir[1], seg_dir[0])
        waypoints = np.array([x[:2] + i * WAYPOINT_SPACING_M * seg_dir for i in range(0, N_WAYPOINTS + 1)])
        chunk_times = np.arange(N_WAYPOINTS + 1) * WAYPOINT_DT
        fine_times = np.arange(0.0, chunk_times[-1] + 1e-9, mpc.dt)
        xs_i = np.interp(fine_times, chunk_times, waypoints[:, 0])
        ys_i = np.interp(fine_times, chunk_times, waypoints[:, 1])
        thetas_i = np.full_like(xs_i, heading)

        # Reproduces mpc_path_follower.py's build_xref(): clip to the chunk's own
        # span, then linearly extrapolate the tail (see that method's docstring).
        raw_idxs = np.arange(mpc.N + 1)
        last_idx = len(fine_times) - 1
        x_ref = np.stack([xs_i[np.clip(raw_idxs, 0, last_idx)], ys_i[np.clip(raw_idxs, 0, last_idx)],
                           thetas_i[np.clip(raw_idxs, 0, last_idx)]], axis=1)
        overflow = raw_idxs > last_idx
        if np.any(overflow) and last_idx >= 1:
            last = np.array([xs_i[last_idx], ys_i[last_idx], thetas_i[last_idx]])
            prev = np.array([xs_i[last_idx - 1], ys_i[last_idx - 1], thetas_i[last_idx - 1]])
            step = last[:2] - prev[:2]
            hd = np.arctan2(step[1], step[0]) if np.linalg.norm(step) > 1e-9 else last[2]
            steps_beyond = (raw_idxs[overflow] - last_idx).astype(float)
            x_ref[overflow, 0] = last[0] + steps_beyond * step[0]
            x_ref[overflow, 1] = last[1] + steps_beyond * step[1]
            x_ref[overflow, 2] = hd

        dx = x_ref[1:, 0] - x_ref[:-1, 0]
        dy = x_ref[1:, 1] - x_ref[:-1, 1]
        v = (dx * np.cos(x_ref[:-1, 2]) + dy * np.sin(x_ref[:-1, 2])) / mpc.dt
        u_ref = np.stack([v, np.zeros(mpc.N)], axis=1)

        patch = esdf.get_patch(x[:2])
        u0, predicted_states, ok = mpc.solve(x, x_ref, u_ref, psi, patch)
        slacks = mpc.extract_slacks(mpc._w_guess)
        max_slack_seen = max(max_slack_seen, float(slacks.max()))
        if not ok:
            n_fail += 1

        if verbose and (tick % 5 == 0 or slacks.max() > 1e-4):
            print(f'tick={tick:3d} x={x[0]:.3f} y={x[1]:.3f} ok={ok} '
                  f'u0=[{u0[0]:+.3f},{u0[1]:+.3f}] max_Sk={slacks.max():.4f}')

        xs.append(x[0])
        x = x + CONTROL_LOOP_DT * np.array([u0[0] * np.cos(x[2]), u0[0] * np.sin(x[2]), u0[1]])

    xs = np.array(xs)
    # Deadlock heuristic: did x stop advancing (last quarter of the run moved < 5cm)?
    last_quarter = xs[3 * len(xs) // 4:]
    stalled = (last_quarter.max() - last_quarter.min()) < 0.05
    print(f'SUMMARY final_x={x[0]:.3f} max_slack={max_slack_seen:.4f} '
          f'n_fail={n_fail}/{n_ticks} stalled={stalled}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ticks', type=int, default=250)
    p.add_argument('--patch-resolution', type=float, default=0.05,
                    help='esdf_single_obs was recorded at 0.05m, not the 0.1m production default')
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--qf-multiplier', type=float, default=None,
                    help='override TERMINAL_COST_Q by this multiple of the base DARE solution (QF)')
    p.add_argument('--horizon-n', type=int, default=None, help='override PREDICTION_HORIZON_N/CONTROL_HORIZON_M')
    p.add_argument('--low-v-weight', type=float, default=None,
                    help='override LOW_V_WEIGHT (soft low-velocity floor, attempt #12)')
    p.add_argument('--low-v-threshold', type=float, default=None, help='override LOW_V_THRESHOLD [m/s]')
    p.add_argument('--path-origin', type=float, nargs=2, default=None, metavar=('X', 'Y'),
                    help='override PATH_ORIGIN - robot starts here too, facing the line heading')
    p.add_argument('--path-heading', type=float, default=None,
                    help='override PATH_HEADING [rad] - robot starts facing this way too')
    args = p.parse_args()
    run(args.ticks, args.patch_resolution, args.verbose, args.qf_multiplier, args.horizon_n,
        args.low_v_weight, args.low_v_threshold, args.path_origin, args.path_heading)
