#!/usr/bin/env python3
"""Simulation loop comparing AsyncFeedForwardController vs PurePursuitController.

Traces each controller's resulting path against the action chunk's own
waypoints, for two test cases (a simple curve and an S-curve), and saves
side-by-side comparison plots. No ROS, no colcon, just the compiled module.
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless - this environment has no display
import matplotlib.pyplot as plt

# The compiled module lands in the CMake build directory next to this script.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))

import action_chunk_controller_py as acc  # noqa: E402

STEP_DISTANCE = 0.12    # matches debug/simulate_action_chunk.py's default
TURN_PER_STEP = 0.08    # matches debug/simulate_action_chunk.py's default
CONTROL_RATE_HZ = 30.0  # matches the real hardware control tick rate
DT = 1.0 / CONTROL_RATE_HZ
DURATION_SEC = 3.0


def naivePositionEstimator(pose, speed_body_x, yaw_rate, dt):
    """Naive forward-Euler step: assumes the commanded velocity is achieved
    immediately, so P = v * dt. Doesn't know or care which controller produced
    the command - same integration applies unchanged for any future controller."""
    x, y, theta = pose
    x += speed_body_x * math.cos(theta) * dt
    y += speed_body_x * math.sin(theta) * dt
    theta += yaw_rate * dt
    return [x, y, theta]


def estimatePosLagged(pose, actual_velocity, commanded_speed_body_x, commanded_yaw_rate, dt):
    """Non-naive position estimator: models actuator lag (velocity doesn't
    reach the commanded value instantly) and wheel slip (achieved velocity
    doesn't fully translate to displacement). Unlike estimatePosPrim, this is
    NOT a pure function of (pose, command, dt) - it needs actual_velocity
    threaded through tick to tick as real simulator state, since "how fast am
    I actually going right now" no longer just equals "what was commanded".

    TODO: implement, roughly:
    - Lag (first-order actuator response), separately for linear and angular:
        actual_v += (commanded_speed_body_x - actual_v) * (dt / tau_v)
        actual_w += (commanded_yaw_rate     - actual_w) * (dt / tau_w)
      tau_v / tau_w are time constants - ~0.1-0.3s is a plausible starting
      guess for a small rover's motor+gearbox response, but needs measuring
      against the real hardware rather than assumed.
    - Slip: scale the post-lag velocity down before integrating position:
        effective_v = actual_v * slip_factor
      slip_factor candidates, roughly increasing in complexity:
        * a fixed constant < 1 (simplest - constant traction loss)
        * a function of |actual_w| (skid-steer/diff-drive turning causes more
          wheel scrubbing than driving straight, so slip should worsen with
          turn rate, not just be constant)
        * randomized per tick (e.g. slip_factor ~ N(0.95, 0.05), clipped to
          [0, 1]) to simulate unpredictable terrain/traction variation
    - Integrate pose from the EFFECTIVE (post-lag, post-slip) velocity using
      the same P = v*dt approach as estimatePosPrim - only which velocity
      feeds that integration differs.
    - Return (new_pose, new_actual_velocity) - the caller has to carry
      actual_velocity forward to the next tick, same as pose.
    """
    raise NotImplementedError


def wrapToPi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def toWorldFrame(chunk, anchor_pose):
    """Rotate + translate a robot-relative chunk into world frame, using the
    pose the robot was actually at when this chunk was generated/anchored."""
    anchor_x, anchor_y, anchor_theta = anchor_pose
    cos_yaw = math.cos(anchor_theta)
    sin_yaw = math.sin(anchor_theta)

    world_chunk = []
    for rel_x, rel_y, rel_theta in chunk:
        world_x = anchor_x + cos_yaw * rel_x - sin_yaw * rel_y
        world_y = anchor_y + sin_yaw * rel_x + cos_yaw * rel_y
        world_theta = wrapToPi(anchor_theta + rel_theta)
        world_chunk.append([world_x, world_y, world_theta])
    return world_chunk


def build_left_arc_chunk(step_distance=STEP_DISTANCE, turn_per_step=TURN_PER_STEP):
    """Mirrors the left_arc pattern in debug/simulate_action_chunk.py's build_pose().
    Constant-curvature arc, turning left throughout - the "simple curve" case."""
    chunk = []
    for index in range(1, 9):
        distance = step_distance * index
        theta = turn_per_step * index
        x = distance * math.cos(0.5 * theta)
        y = distance * math.sin(0.5 * theta)
        chunk.append([x, y, theta])
    return chunk


def build_s_curve_chunk(step_distance=STEP_DISTANCE, turn_per_step=TURN_PER_STEP):
    """Turns left for the first half of the chunk, then right for the second
    half - an S-shape, unlike build_left_arc_chunk's constant-curvature arc.
    Built by forward-integrating heading + position step by step, since the
    closed-form arc formula only holds for constant curvature."""
    chunk = []
    x, y, theta = 0.0, 0.0, 0.0
    for index in range(1, 9):
        direction = 1.0 if index <= 4 else -1.0
        theta += direction * turn_per_step
        x += step_distance * math.cos(theta)
        y += step_distance * math.sin(theta)
        chunk.append([x, y, theta])
    return chunk


def build_right_arc_chunk(step_distance=STEP_DISTANCE, turn_per_step=TURN_PER_STEP):
    """Same construction as build_left_arc_chunk, curving the other way."""
    return build_left_arc_chunk(step_distance=step_distance, turn_per_step=-turn_per_step)


def run_simulation(command_fn, chunk, anchor_pose):
    """Drive naivePositionEstimator forward using whatever command_fn (bound
    to one controller instance) returns each tick."""
    pose = list(anchor_pose)
    trajectory = [list(pose)]

    num_ticks = round(DURATION_SEC / DT)
    for _ in range(num_ticks):
        speed_body_x, yaw_rate = command_fn(chunk, pose)
        pose = naivePositionEstimator(pose, speed_body_x, yaw_rate, DT)
        trajectory.append(list(pose))
    return trajectory


def run_simulation_multi_chunk(command_fn, chunks, ticks_per_chunk):
    """Like run_simulation, but swaps to the next chunk partway through -
    simulating a new VLA inference replacing the in-flight chunk, each with
    its own seq_num (the chunk index) so PurePursuitController re-anchors.
    Each chunk is anchored to wherever the robot has actually reached by the
    time it becomes active, not a shared/fixed schedule - this differs between
    controllers since their trajectories diverge.

    Returns (trajectory, world_chunks, swap_indices): world_chunks are each
    chunk's own waypoints transformed into world frame using its real anchor;
    swap_indices are indices into trajectory where each chunk became active.
    """
    pose = [0.0, 0.0, 0.0]
    trajectory = [list(pose)]
    world_chunks = []
    swap_indices = []

    for chunk_index, chunk in enumerate(chunks):
        world_chunks.append(toWorldFrame(chunk, pose))
        swap_indices.append(len(trajectory) - 1)

        for _ in range(ticks_per_chunk):
            speed_body_x, yaw_rate = command_fn(chunk, chunk_index, pose)
            pose = naivePositionEstimator(pose, speed_body_x, yaw_rate, DT)
            trajectory.append(list(pose))

    return trajectory, world_chunks, swap_indices


def plot_comparison_multi_chunk(title, results, out_path):
    """results: list of (name, trajectory, world_chunks, swap_indices) tuples,
    one per controller. Shows every chunk actually sent during the run, each
    in its own color, plus the trajectory and a marker where each swap hit."""
    chunk_colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, traj, world_chunks, swap_indices) in zip(axes, results):
        traj_x = [p[0] for p in traj]
        traj_y = [p[1] for p in traj]
        ax.plot(traj_x, traj_y, "-", color="tab:blue", label="simulated trajectory")

        for i, world_chunk in enumerate(world_chunks):
            wx = [wp[0] for wp in world_chunk]
            wy = [wp[1] for wp in world_chunk]
            color = chunk_colors[i % len(chunk_colors)]
            ax.plot(wx, wy, "o--", color=color, label=f"chunk {i + 1} waypoints")

        swap_x = [traj[i][0] for i in swap_indices]
        swap_y = [traj[i][1] for i in swap_indices]
        ax.scatter(swap_x, swap_y, color="black", marker="x", s=80, zorder=6,
                   label="chunk swap point")

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(name)
        ax.axis("equal")
        ax.grid(True)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_case_chunk_replacement(title, chunks, ticks_per_chunk, out_path):
    async_controller = acc.AsyncFeedForwardController()
    pp_controller = acc.PurePursuitController()

    async_fn = lambda c, seq, p: async_controller.compute_command(c, p)  # noqa: E731
    pp_fn = lambda c, seq, p: pp_controller.compute_command(c, seq, p)  # noqa: E731

    async_traj, async_world_chunks, async_swaps = run_simulation_multi_chunk(
        async_fn, chunks, ticks_per_chunk
    )
    pp_traj, pp_world_chunks, pp_swaps = run_simulation_multi_chunk(
        pp_fn, chunks, ticks_per_chunk
    )

    plot_comparison_multi_chunk(
        title,
        [
            ("AsyncFeedForwardController", async_traj, async_world_chunks, async_swaps),
            ("PurePursuitController", pp_traj, pp_world_chunks, pp_swaps),
        ],
        out_path,
    )


def plot_comparison(title, world_chunk, async_traj, pp_traj, out_path):
    """Side-by-side subplots: AsyncFeedForwardController vs PurePursuitController,
    both traced against the same action chunk waypoints."""
    chunk_x = [wp[0] for wp in world_chunk]
    chunk_y = [wp[1] for wp in world_chunk]

    fig, (ax_async, ax_pp) = plt.subplots(1, 2, figsize=(12, 6))

    for ax, traj, name in (
        (ax_async, async_traj, "AsyncFeedForwardController"),
        (ax_pp, pp_traj, "PurePursuitController"),
    ):
        traj_x = [p[0] for p in traj]
        traj_y = [p[1] for p in traj]
        ax.plot(chunk_x, chunk_y, "o--", color="tab:orange", label="action chunk waypoints")
        ax.plot(traj_x, traj_y, "-", color="tab:blue", label="simulated trajectory")
        ax.scatter([0], [0], color="black", zorder=5, label="start")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(name)
        ax.axis("equal")
        ax.grid(True)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_case(title, chunk, out_path):
    anchor_pose = [0.0, 0.0, 0.0]
    world_chunk = toWorldFrame(chunk, anchor_pose)

    async_controller = acc.AsyncFeedForwardController()
    pp_controller = acc.PurePursuitController()

    async_traj = run_simulation(
        lambda c, p: async_controller.compute_command(c, p), chunk, anchor_pose
    )
    # Fixed seq_num throughout - it's the same chunk for the whole simulated
    # run, so PurePursuitController should only (re)generate waypoints once,
    # on the very first tick.
    pp_traj = run_simulation(
        lambda c, p: pp_controller.compute_command(c, 1, p), chunk, anchor_pose
    )

    plot_comparison(title, world_chunk, async_traj, pp_traj, out_path)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    run_case(
        "Case 1: Simple Curve",
        build_left_arc_chunk(),
        os.path.join(here, "comparison_simple_curve.png"),
    )
    run_case(
        "Case 2: S-Curve",
        build_s_curve_chunk(),
        os.path.join(here, "comparison_s_curve.png"),
    )
    run_case_chunk_replacement(
        "Case 3: Action Chunk Replacement (arc left, then chunk swapped to arc right)",
        [build_left_arc_chunk(), build_right_arc_chunk()],
        ticks_per_chunk=round((DURATION_SEC / 2) / DT),
        out_path=os.path.join(here, "comparison_chunk_replacement.png"),
    )


if __name__ == "__main__":
    main()
