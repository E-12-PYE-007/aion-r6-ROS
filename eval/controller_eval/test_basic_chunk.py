#!/usr/bin/env python3
"""Simulation loop for the AsyncFeedForwardController Python binding.

Traces the resulting path against the action chunk's own waypoints for a
left-curving chunk. No ROS, no colcon, just the compiled module.
"""

import math
import os
import sys

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
    pose the robot was actually at when this chunk was generated/anchored.
    Same transform as OdometryPurePursuitController::setActionChunk (C++) -
    needed the moment a chunk isn't anchored at the origin, e.g. once chunks
    get regenerated mid-loop at wherever the robot has actually moved to."""
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
    """Mirrors the left_arc pattern in debug/simulate_action_chunk.py's build_pose()."""
    chunk = []
    for index in range(1, 9):
        distance = step_distance * index
        theta = turn_per_step * index
        x = distance * math.cos(0.5 * theta)
        y = distance * math.sin(0.5 * theta)
        chunk.append([x, y, theta])
    return chunk


def main():
    controller = acc.AsyncFeedForwardController()  # swap this line for a different controller later
    chunk = build_left_arc_chunk()
    anchor_pose = [0.0, 0.0, 0.0]  # pose the robot was at when this chunk was generated
    pose = list(anchor_pose)

    trajectory = [list(pose)]

    num_ticks = round(DURATION_SEC / DT)
    for _ in range(num_ticks):
        speed_body_x, yaw_rate = controller.compute_command(chunk, pose)
        pose = naivePositionEstimator(pose, speed_body_x, yaw_rate, DT)
        trajectory.append(list(pose))

    world_chunk = toWorldFrame(chunk, anchor_pose)
    plot(world_chunk, trajectory)


def plot(world_chunk, trajectory):
    chunk_x = [wp[0] for wp in world_chunk]
    chunk_y = [wp[1] for wp in world_chunk]
    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]

    plt.figure()
    plt.plot(chunk_x, chunk_y, "o--", color="tab:orange", label="action chunk waypoints")
    plt.plot(traj_x, traj_y, "-", color="tab:blue", label="simulated trajectory")
    plt.scatter([0], [0], color="black", zorder=5, label="start")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("AsyncFeedForwardController: traced path vs. action chunk")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
