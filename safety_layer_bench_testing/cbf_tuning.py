#!/usr/bin/env python3
"""Offline closed-loop harness for tuning the CBF safety layer. No launch, no waiting: a run
takes about half a second.

It replays what the bench does - the sim robot (50 Hz unicycle), the chunk generator (8 Hz,
goal 6 m ahead of the start), and the CBF safety layer (15 Hz: pure-pursuit nominal, then
the CBF-QP) - using the same code as the nodes, against the bench ESDF. Needs the workspace
sourced (control, debug and aion_msgs).

  python3 safety_layer_bench_testing/cbf_tuning.py                        # repo values
  python3 safety_layer_bench_testing/cbf_tuning.py --set alpha=1.5
  python3 safety_layer_bench_testing/cbf_tuning.py --sweep alpha=1,2,3 --sweep lookahead=0.25,0.4
  python3 safety_layer_bench_testing/cbf_tuning.py --set extra_margin=0.2 --k-curv 0.5

Knobs for --set / --sweep: alpha, lookahead, weight_v, weight_omega, slack_weight,
extra_margin (added to SAFETY_MARGIN_M, as CBF_EXTRA_MARGIN_M is). --k-curv overrides
PP_K_CURV, the pure-pursuit curvature slowdown.

Each variant is run from every start pose and summarised over them:
  inside%       share of the run the robot's CENTRE is within --margin of an obstacle (worst run)
  min centre    smallest ESDF distance at the centre over all runs
  min h         smallest barrier value the filter reported (negative = barrier violated)
  max slack     largest CBF slack used
  reached       runs that end within 0.6 m of the goal
  t goal        mean time to come within 0.5 m of the goal, over the runs that get there
"""
import argparse
import itertools
import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_run  # noqa: E402  (ESDF reader shared with the plotter)

from aion_msgs.msg import ActionChunk  # noqa: E402
from control.safety_layer.cbf.cbf_qp import CbfFilter  # noqa: E402
from control.safety_layer.cbf.constants import CBF_EXTRA_MARGIN_M  # noqa: E402
from control.safety_layer.common import pure_pursuit  # noqa: E402
from control.safety_layer.common.constants import SAFETY_MARGIN_M  # noqa: E402
from control.safety_layer.common.esdf_map import EsdfMap  # noqa: E402
from debug.chunk_generator import chunk_toward_goal  # noqa: E402
from geometry_msgs.msg import Pose2D  # noqa: E402

DEFAULT_BAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'rosbags', 'esdf_single_obs', 'esdf_single_obs_0.db3')
PATCH_SIZE, PATCH_RESOLUTION = 44, 0.05   # the bench ESDF's own resolution

SIM_DT, CONTROL_DT, CHUNK_DT = 0.02, 1 / 15, 1 / 8
GOAL_DISTANCE = 6.0
FIRST_CONTROL_S = 0.5                      # the node starts once odometry and a chunk exist

# (x, y, heading): y offsets across the obstacle at heading 0, then angled and offset starts.
START_POSES = [(0.25, y, 0.0) for y in (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6)] + [
    (0.25, -0.5, 0.3), (0.25, 0.5, -0.3), (0.25, -0.3, 0.2), (0.25, 0.3, -0.2),
    (0.25, -0.8, 0.45), (0.25, 0.8, -0.45), (0.25, 0.0, 0.15), (0.25, 0.0, -0.15)]


class Bench:
    def __init__(self, bag):
        self.esdf = plot_run.Esdf(bag)
        raw = np.where(self.esdf.unknown, 1000.0, self.esdf.values)
        self.msg = types.SimpleNamespace(
            data=raw.ravel().tolist(), height=raw.shape[0], width=raw.shape[1], unknown_value=1000.0,
            origin=types.SimpleNamespace(x=self.esdf.origin[0], y=self.esdf.origin[1]),
            resolution=self.esdf.resolution, header=types.SimpleNamespace(frame_id='odom'))

    def run(self, cbf, start, duration):
        """One closed-loop run. Returns an (n, 9) array: t, x, y, theta, v, omega, slack, h, ESDF at centre."""
        esdf_map = EsdfMap(PATCH_SIZE)
        esdf_map.store(self.msg)
        follower = pure_pursuit.PurePursuit()
        pose = np.array(start, dtype=float)
        goal = pose[:2] + GOAL_DISTANCE * np.array([math.cos(pose[2]), math.sin(pose[2])])
        cmd, slack, h, chunk, seq = np.zeros(2), 0.0, 0.0, None, 0
        t, next_control, next_chunk = 0.0, FIRST_CONTROL_S, 0.0
        history = []
        while t < duration:
            if t >= next_chunk:
                chunk = ActionChunk()
                chunk.seq_num, seq = seq, seq + 1
                for i, (x, y, theta) in enumerate(chunk_toward_goal(pose, goal)):
                    chunk.relative_poses[i] = Pose2D(x=x, y=y, theta=theta)
                next_chunk += CHUNK_DT
            if t >= next_control and chunk is not None:
                v_nom, omega_nom = follower.nominal(chunk, pose.copy())
                u, h, slack = cbf.filter(pose, 0.0, esdf_map.get_patch(pose[:2]), [v_nom, omega_nom])
                cmd = np.array(u, dtype=float)
                next_control += CONTROL_DT
            pose = pose + SIM_DT * np.array([cmd[0] * math.cos(pose[2]), cmd[0] * math.sin(pose[2]), cmd[1]])
            t += SIM_DT
            history.append((t, pose[0], pose[1], pose[2], cmd[0], cmd[1], slack, h,
                            float(self.esdf.distance_at(pose[0], pose[1]))))
        return np.array(history), goal


def summarise(runs, margin):
    inside, min_centre, min_h, max_slack, reached, t_goal = [], [], [], 0.0, 0, []
    for history, goal in runs:
        d = history[:, 8]
        valid = ~np.isnan(d)
        inside.append(100.0 * float((d[valid] < margin).mean()))
        min_centre.append(float(np.nanmin(d)))
        moving = np.flatnonzero(history[:, 4] > 0.01)
        min_h.append(float(history[moving[0]:, 7].min()) if len(moving) else 0.0)
        max_slack = max(max_slack, float(history[:, 6].max()))
        dist = np.hypot(goal[0] - history[:, 1], goal[1] - history[:, 2])
        reached += int(dist[-1] < 0.6)
        near = np.flatnonzero(dist < 0.5)
        if len(near):
            t_goal.append(float(history[near[0], 0]))
    return dict(inside=max(inside), min_centre=min(min_centre), min_h=min(min_h), max_slack=max_slack,
                reached=reached, n=len(runs), t_goal=float(np.mean(t_goal)) if t_goal else float('nan'),
                n_t_goal=len(t_goal))


def make_filter(settings):
    settings = dict(settings)
    extra = settings.pop('extra_margin', CBF_EXTRA_MARGIN_M)
    return CbfFilter(patch_size=PATCH_SIZE, patch_resolution=PATCH_RESOLUTION,
                     safety_margin=SAFETY_MARGIN_M + extra, **settings)


def parse_values(text):
    return [float(v) for v in text.split(',')]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', action='append', default=[], metavar='KNOB=VALUE', help='fix a knob for every variant')
    parser.add_argument('--sweep', action='append', default=[], metavar='KNOB=V1,V2,...', help='run every combination')
    parser.add_argument('--k-curv', type=float, help='override PP_K_CURV')
    parser.add_argument('--duration', type=float, default=30.0, help='run length [s] (default 30)')
    parser.add_argument('--margin', type=float, default=SAFETY_MARGIN_M, help='margin the centre is judged against')
    parser.add_argument('--bag', default=DEFAULT_BAG, help='ESDF rosbag')
    args = parser.parse_args()

    if args.k_curv is not None:
        pure_pursuit.PP_K_CURV = args.k_curv
    fixed = {k: float(v) for k, v in (item.split('=') for item in args.set)}
    sweeps = [(k, parse_values(v)) for k, v in (item.split('=') for item in args.sweep)]
    names = [k for k, _ in sweeps]

    bench = Bench(args.bag)
    print(f'{len(START_POSES)} start poses, {args.duration:.0f} s each; centre judged against {args.margin:.2f} m')
    header = ' '.join(f'{n:>12}' for n in names)
    print(f'{header}{" " if names else ""}| inside%  min centre  min h    max slack  reached  t goal')
    for combo in itertools.product(*[values for _, values in sweeps]) if sweeps else [()]:
        settings = dict(fixed, **dict(zip(names, combo)))
        cbf = make_filter(settings)
        summary = summarise([bench.run(cbf, s, args.duration) for s in START_POSES], args.margin)
        label = ' '.join(f'{v:>12g}' for v in combo)
        print(f'{label}{" " if names else ""}| {summary["inside"]:6.1f}   {summary["min_centre"]:9.3f}  '
              f'{summary["min_h"]:+7.3f}  {summary["max_slack"]:9.3f}  {summary["reached"]:>3}/{summary["n"]:<3}  '
              f'{summary["t_goal"]:5.1f} s ({summary["n_t_goal"]} of {summary["n"]})')


if __name__ == '__main__':
    main()
