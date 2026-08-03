#!/usr/bin/env python3
"""Simple path post-processing for expert action chunk generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sim.expert_trajectory_utils import path_length, sample_path_pose, segment_lengths, wrap_to_pi


@dataclass
class TimedTrajectory:
    path: list[np.ndarray]
    times: np.ndarray
    distances: np.ndarray
    yaws: np.ndarray
    speeds: np.ndarray
    yaw_rates: np.ndarray

    def duration(self) -> float:
        if len(self.times) == 0:
            return 0.0
        return float(self.times[-1])

    def sample(self, query_time: float) -> tuple[np.ndarray, float, float, float]:
        if len(self.path) == 0:
            raise ValueError("Cannot sample empty trajectory.")
        query_time = min(max(query_time, 0.0), self.duration())
        distance = float(np.interp(query_time, self.times, self.distances))
        speed = float(np.interp(query_time, self.times, self.speeds))
        yaw_rate = float(np.interp(query_time, self.times, self.yaw_rates))
        position, yaw = sample_path_pose(self.path, distance)
        return position, yaw, speed, yaw_rate


def resample_path(path: list[np.ndarray], spacing_m: float) -> list[np.ndarray]:
    total = path_length(path)
    if len(path) < 2 or total <= 1e-9:
        return path
    spacing_m = max(spacing_m, 1e-3)
    count = max(2, int(total / spacing_m) + 1)
    return [sample_path_pose(path, i * total / (count - 1))[0] for i in range(count)]


def build_timed_trajectory(
    path: list[np.ndarray],
    max_speed_mps: float,
    max_yaw_rate_radps: float,
    max_accel_mps2: float,
    max_decel_mps2: float,
    max_angular_accel_radps2: float,
    min_speed_mps: float = 0.03,
    stop_at_end: bool = True,
) -> TimedTrajectory:
    if len(path) < 2:
        zeros = np.zeros(1, dtype=np.float64)
        return TimedTrajectory(path, zeros, zeros, zeros, zeros, zeros)

    lengths = segment_lengths(path)
    distances = np.concatenate(([0.0], np.cumsum(lengths)))
    yaws = path_yaws(path)
    curvature = estimate_curvature(distances, yaws)

    speeds = np.full(len(path), float(max_speed_mps), dtype=np.float64)
    for i, kappa in enumerate(curvature):
        if abs(kappa) > 1e-6:
            speeds[i] = min(speeds[i], float(max_yaw_rate_radps) / abs(kappa))
    speeds = np.maximum(speeds, min(float(min_speed_mps), float(max_speed_mps)))
    speeds[0] = min(speeds[0], min_speed_mps)
    if stop_at_end:
        speeds[-1] = 0.0

    for i in range(1, len(speeds)):
        ds = max(distances[i] - distances[i - 1], 0.0)
        speeds[i] = min(speeds[i], math.sqrt(max(speeds[i - 1] ** 2 + 2.0 * max_accel_mps2 * ds, 0.0)))

    for i in range(len(speeds) - 2, -1, -1):
        ds = max(distances[i + 1] - distances[i], 0.0)
        speeds[i] = min(speeds[i], math.sqrt(max(speeds[i + 1] ** 2 + 2.0 * max_decel_mps2 * ds, 0.0)))

    times = np.zeros(len(path), dtype=np.float64)
    for i in range(1, len(path)):
        ds = max(distances[i] - distances[i - 1], 0.0)
        avg_speed = max((speeds[i - 1] + speeds[i]) * 0.5, min_speed_mps)
        times[i] = times[i - 1] + ds / avg_speed

    times = enforce_yaw_rate_limit(times, yaws, max_yaw_rate_radps)
    yaw_rates = compute_yaw_rates(times, yaws)
    times, yaw_rates = enforce_angular_accel_limit(times, yaws, yaw_rates, max_angular_accel_radps2)
    speeds = effective_speeds_from_times(distances, times, stop_at_end)
    return TimedTrajectory(path, times, distances, yaws, speeds, yaw_rates)


def path_yaws(path: list[np.ndarray]) -> np.ndarray:
    yaws = np.zeros(len(path), dtype=np.float64)
    for i in range(len(path)):
        if i == len(path) - 1:
            direction = path[i] - path[i - 1]
        else:
            direction = path[i + 1] - path[i]
        yaws[i] = math.atan2(float(direction[1]), float(direction[0]))
    return unwrap_angles(yaws)


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    if len(angles) == 0:
        return angles
    unwrapped = np.asarray(angles, dtype=np.float64).copy()
    for i in range(1, len(unwrapped)):
        unwrapped[i] = unwrapped[i - 1] + wrap_to_pi(float(unwrapped[i]) - float(unwrapped[i - 1]))
    return unwrapped


def estimate_curvature(distances: np.ndarray, yaws: np.ndarray) -> np.ndarray:
    curvature = np.zeros(len(yaws), dtype=np.float64)
    for i in range(1, len(yaws) - 1):
        ds = max(float(distances[i + 1] - distances[i - 1]), 1e-6)
        curvature[i] = wrap_to_pi(float(yaws[i + 1]) - float(yaws[i - 1])) / ds
    if len(yaws) > 2:
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
    return curvature


def compute_yaw_rates(times: np.ndarray, yaws: np.ndarray) -> np.ndarray:
    yaw_rates = np.zeros(len(yaws), dtype=np.float64)
    for i in range(1, len(yaws)):
        dt = max(float(times[i] - times[i - 1]), 1e-6)
        yaw_rates[i] = wrap_to_pi(float(yaws[i]) - float(yaws[i - 1])) / dt
    return yaw_rates


def enforce_yaw_rate_limit(
    times: np.ndarray,
    yaws: np.ndarray,
    max_yaw_rate_radps: float,
) -> np.ndarray:
    if max_yaw_rate_radps <= 0.0 or len(times) < 2:
        return times
    adjusted = np.asarray(times, dtype=np.float64).copy()
    for i in range(1, len(adjusted)):
        yaw_delta = abs(wrap_to_pi(float(yaws[i]) - float(yaws[i - 1])))
        required_dt = yaw_delta / max(max_yaw_rate_radps, 1e-6)
        actual_dt = max(float(adjusted[i] - adjusted[i - 1]), 1e-6)
        if required_dt > actual_dt:
            adjusted[i:] += required_dt - actual_dt
    return adjusted


def effective_speeds_from_times(distances: np.ndarray, times: np.ndarray, stop_at_end: bool) -> np.ndarray:
    speeds = np.zeros(len(times), dtype=np.float64)
    if len(times) < 2:
        return speeds
    segment_speeds = np.zeros(len(times) - 1, dtype=np.float64)
    for i in range(1, len(times)):
        dt = max(float(times[i] - times[i - 1]), 1e-6)
        ds = max(float(distances[i] - distances[i - 1]), 0.0)
        segment_speeds[i - 1] = ds / dt
    speeds[0] = segment_speeds[0]
    for i in range(1, len(times) - 1):
        speeds[i] = min(segment_speeds[i - 1], segment_speeds[i])
    speeds[-1] = 0.0 if stop_at_end else segment_speeds[-1]
    return speeds


def enforce_angular_accel_limit(
    times: np.ndarray,
    yaws: np.ndarray,
    yaw_rates: np.ndarray,
    max_angular_accel_radps2: float,
) -> tuple[np.ndarray, np.ndarray]:
    if max_angular_accel_radps2 <= 0.0 or len(times) < 2:
        return times, yaw_rates
    adjusted = np.asarray(times, dtype=np.float64).copy()
    for _ in range(2):
        yaw_rates = compute_yaw_rates(adjusted, yaws)
        for i in range(1, len(adjusted)):
            rate_delta = abs(float(yaw_rates[i]) - float(yaw_rates[i - 1]))
            required_dt = rate_delta / max(max_angular_accel_radps2, 1e-6)
            actual_dt = max(float(adjusted[i] - adjusted[i - 1]), 1e-6)
            if required_dt > actual_dt:
                adjusted[i:] += required_dt - actual_dt
    return adjusted, compute_yaw_rates(adjusted, yaws)


def shortcut_smooth(path: list[np.ndarray], collision_fn, iterations: int = 2) -> list[np.ndarray]:
    if len(path) <= 2:
        return path
    smoothed = list(path)
    for _ in range(iterations):
        output = [smoothed[0]]
        index = 0
        while index < len(smoothed) - 1:
            next_index = len(smoothed) - 1
            while next_index > index + 1:
                if segment_is_free(smoothed[index], smoothed[next_index], collision_fn):
                    break
                next_index -= 1
            output.append(smoothed[next_index])
            index = next_index
        smoothed = output
    return smoothed


def segment_is_free(start: np.ndarray, end: np.ndarray, collision_fn, step_m: float = 0.1) -> bool:
    distance = float(np.linalg.norm(end - start))
    steps = max(2, int(distance / step_m) + 1)
    for i in range(steps + 1):
        point = start + (end - start) * (i / steps)
        if collision_fn(point):
            return False
    return True
