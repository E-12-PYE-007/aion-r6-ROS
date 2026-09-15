import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import correlate
from scipy.signal import correlation_lags

from pathlib import Path

import rosbag2_py

from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

BAG = "mocap_pilot_rotation_0.4/t1"
BAG_PATH = "rosbag/" + BAG

MOCAP_TOPIC = "/mocap/rover/pose"
WHEEL_TOPIC = "/odometry/wheel"

def read_data():
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=BAG_PATH,
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    reader.set_filter(
        rosbag2_py.StorageFilter(topics=[MOCAP_TOPIC, WHEEL_TOPIC])
    )

    mocap_data = {}
    mocap_data['timestamps'] = []
    mocap_data['positions'] = {'x': [], 'y': [], 'z': []}
    mocap_data['orientations'] = {'x': [], 'y': [], 'z': [], 'w': []}
    wheel_data = {}
    wheel_data['timestamps'] = []
    wheel_data['velocities'] = {'linear': {'x': [], 'y': []}, 'angular': {'z': []}}
    first_timestamp = None

    while reader.has_next():
        topic, serialized_data, timestamp = reader.read_next()

        if first_timestamp is None:
            first_timestamp = timestamp

        # time since start of bag in seconds
        time_s = (timestamp - first_timestamp) / 1e9

        if topic == MOCAP_TOPIC:
            msg = deserialize_message(serialized_data, PoseStamped)
            p = msg.pose.position
            q = msg.pose.orientation

            mocap_data['timestamps'].append(time_s)
            mocap_data['positions']['x'].append(p.x)
            mocap_data['positions']['y'].append(p.y)
            mocap_data['positions']['z'].append(p.z)
            mocap_data['orientations']['x'].append(q.x)
            mocap_data['orientations']['y'].append(q.y)
            mocap_data['orientations']['z'].append(q.z)
            mocap_data['orientations']['w'].append(q.w)

        elif topic == WHEEL_TOPIC:
            msg = deserialize_message(serialized_data, Odometry)
            velocity = msg.twist.twist

            wheel_data['timestamps'].append(time_s)
            wheel_data['velocities']['linear']['x'].append(velocity.linear.x)
            wheel_data['velocities']['linear']['y'].append(velocity.linear.y)
            wheel_data['velocities']['angular']['z'].append(velocity.angular.z)

    return mocap_data, wheel_data

def plot_mocap_and_wheel_data(mocap_data, wheel_data):
    mocap_velocities = np.diff(mocap_data['positions']['x'], axis=0) / np.diff(mocap_data['timestamps'], axis=0)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(mocap_data['timestamps'], mocap_data['positions']['x'], label="Mocap X")
    plt.plot(mocap_data['timestamps'], mocap_data['positions']['y'], label="Mocap Y")
    plt.plot(mocap_data['timestamps'], mocap_data['positions']['z'], label="Mocap Z")
    plt.title("Mocap Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(mocap_data['timestamps'][1:], mocap_velocities, label="Mocap Velocity X")
    plt.plot(mocap_data['timestamps'][1:], mocap_velocities, label="Mocap Velocity Y")
    plt.title("Mocap Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(wheel_data['timestamps'], wheel_data['velocities']['linear']['x'], label="Wheel Linear X")
    plt.plot(wheel_data['timestamps'], wheel_data['velocities']['linear']['y'], label="Wheel Linear Y")
    # plt.plot(wheel_data['timestamps'], wheel_data['velocities']['angular']['z'], label="Wheel Angular Z")
    plt.title("Wheel Odometry")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s or rad/s)")
    plt.legend()
    plt.show()

def plot_resultant_velocities(mocap_data, wheel_data):
    dt = np.mean(np.diff(mocap_data['timestamps']))

    # Savgol parameters: window length, polynomial order
    window_length = 30
    polyorder = 3
    print("\nSavitzky-Golay Filter Parameters:")
    print(f"dt: {dt:.4f} s, window_length: {window_length}, polyorder: {polyorder}")
    print(f"window_length * dt: {window_length * dt:.4f} s, polyorder * dt: {polyorder * dt:.4f} s")

    # compute velocities using differentiation
    mocap_velocities = np.diff(mocap_data['positions']['x'], axis=0) / np.diff(mocap_data['timestamps'], axis=0)
    mocap_resultant_velocity = np.linalg.norm(mocap_velocities, axis=1)
    wheel_resultant_velocity = np.linalg.norm([
        wheel_data['velocities']['linear']['x'],
        wheel_data['velocities']['linear']['y']
    ], axis=0)

    # compute smooth velocities using Savitzky-Golay filter
    mocap_velocities_smooth = savgol_filter(mocap_velocities, window_length, polyorder, axis=0)
    mocap_resultant_velocity_smooth = np.linalg.norm(mocap_velocities_smooth, axis=1)

    plt.figure()
    plt.plot(mocap_data['timestamps'][1:], mocap_resultant_velocity, label="Mocap Resultant Velocity")
    plt.plot(wheel_data['timestamps'], wheel_resultant_velocity, label="Wheel Resultant Velocity")
    plt.plot(mocap_data['timestamps'][1:], mocap_resultant_velocity_smooth, label="Mocap Resultant Velocity (Smooth)")
    plt.title("Resultant Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.show()

def plot_aligned_velocities_and_errors(aligned_time, aligned_wheel_yaw_rate, aligned_mocap_yaw_rate, filename=None):
    plt.figure()
    plt.plot(aligned_time, aligned_wheel_yaw_rate, label="Aligned Wheel Yaw Rate")
    plt.plot(aligned_time, aligned_mocap_yaw_rate, label="Aligned Mocap Yaw Rate")
    plt.title("Aligned Yaw Rates")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw Rate (rad/s)")
    plt.legend()
    
    if filename:
        plt.savefig(filename)
    plt.show()

def time_sync_ang_velocities(mocap_data, wheel_data):
    # ---------------------------------------------
    # resample position data
    # ---------------------------------------------
    time = mocap_data['timestamps']
    orientation = mocap_data['orientations']

    q = np.column_stack([
        orientation['x'],
        orientation['y'],
        orientation['z'],
        orientation['w']
    ])

    # normalize the quaternions and extract heading (yaw) angle
    norm = np.linalg.norm(q, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-8):
        raise ValueError("Invalid mocap quaternion")

    q = q/norm[:, None]
    qx, qy, qz, qw = q.T

    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))

    # wrap the yaw angle to avoid discontinuities
    yaw = np.unwrap(yaw)

    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValueError("Mocap timestamps must increase")

    dt = np.median(intervals)
    uniform_time = np.arange(time[0], time[-1], dt)
    uniform_yaw = np.interp(uniform_time, time, yaw)



    # savgol parameters:
    smooth_duration = 0.3
    # need odd number of windows for polynomial order 3.
    window_length = int(round(smooth_duration / dt)) + 1
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(window_length, 5)

    polyorder = 3

    print("Window samples:", window_length)
    print("Actual window span:", (window_length - 1) * dt, "seconds")


    mocap_yaw_rate_smooth = savgol_filter(
        uniform_yaw,
        window_length,
        polyorder,
        deriv=1,
        delta=dt,
        axis=0
    )

    wheel_yaw_rate = np.asarray(
      wheel_data['velocities']['angular']['z']
    )

    wheel_time = wheel_data["timestamps"]

    start = max(uniform_time[0], wheel_time[0])
    end = min(uniform_time[-1], wheel_time[-1])
    t_common = np.arange(start, end, dt)

    mocap_interp = np.interp(
        t_common, uniform_time, mocap_yaw_rate_smooth
    )
    wheel_interp = np.interp(
        t_common, wheel_time, wheel_yaw_rate
    )


    # zero mean the signals to focus on the shape of the signals rather than their absolute values
    mocap_centered = mocap_interp - np.mean(mocap_interp)
    wheel_centered = wheel_interp - np.mean(wheel_interp)

    correlation = correlate(mocap_centered, wheel_centered, mode='full')
    lags = correlation_lags(len(mocap_centered), len(wheel_centered), mode='full')

    max_delay = 1.0
    allowed = np.abs(lags * dt) <= max_delay

    best_lag = lags[allowed][np.argmax(correlation[allowed])]
    time_delay = best_lag * dt
    print(f"\nBest lag (in samples): {best_lag}")
    print(f"MoCap arrival data lags encoder data by: {time_delay:.4f} seconds")

    wheel_time = np.asarray(wheel_data['timestamps'])

    # Each velocity message describes the preceding encoder interval.
    comparison_time = (wheel_time[:-1] + wheel_time[1:]) / 2
    wheel_speed = wheel_yaw_rate[1:]

    query_time = comparison_time + time_delay

    valid = (
        (query_time >= uniform_time[0])
        & (query_time <= uniform_time[-1])
    )

    aligned_time = comparison_time[valid]
    aligned_wheel_yaw_rate = wheel_speed[valid]
    aligned_mocap_yaw_rate = np.interp(
        query_time[valid],
        uniform_time,
        mocap_yaw_rate_smooth,
    )

    return (
        aligned_time,
        aligned_wheel_yaw_rate,
        aligned_mocap_yaw_rate,
    )


def main():
    # create output directory for figures
    output_dir = Path("figs") / BAG
    output_dir.mkdir(parents=True, exist_ok=True)



    mocap_data, wheel_data = read_data()

    # plot_mocap_and_wheel_data(mocap_data, wheel_data)
    # plot_resultant_velocities(mocap_data, wheel_data)

    time = np.asarray(mocap_data['timestamps'])
    intervals = np.diff(time)
    
    plt.plot(time[1:], intervals)
    plt.xlabel("Recorder-relative time (s)")
    plt.ylabel("Mocap arrival interval (s)")
    plt.title("Mocap arrival intervals")
    plt.savefig(output_dir / "mocap_arrival_intervals.png")
    plt.show()

    print("Interval percentiles:", np.percentile(intervals, [1, 50, 99]))


    aligned_time, aligned_wheel_yaw_rate, aligned_mocap_yaw_rate = (time_sync_ang_velocities(mocap_data, wheel_data))

    plot_aligned_velocities_and_errors(aligned_time, aligned_wheel_yaw_rate, aligned_mocap_yaw_rate, filename=output_dir / "aligned_velocities.png")


    # Calculate covariance of the yaw rate errors to estimate the R covariance matrix entry for yaw rate
    yaw_rate_errors = aligned_wheel_yaw_rate - aligned_mocap_yaw_rate
    R_yaw_rate = np.var(yaw_rate_errors)    


    print(f"\nEKF R covariance matrix entry for yaw rate: {R_yaw_rate:.10g} (rad/s)^2")
    print(f"Mean yaw rate error: {np.mean(yaw_rate_errors):.10g} rad/s")
    print(f"Variance of yaw rate error: {np.var(yaw_rate_errors):.10g} (rad/s)^2")
    print(f"Standard deviation of yaw rate error: {np.std(yaw_rate_errors):.10g} rad/s")
    print(f"Max yaw rate error: {np.max(yaw_rate_errors):.10g} rad/s")
    print(f"Min yaw rate error: {np.min(yaw_rate_errors):.10g} rad/s")

    motion_start = 16.0
    motion_end = 24.0

    steady = (aligned_time >= motion_start) & (aligned_time <= motion_end)
    steady_yaw_rate_errors = yaw_rate_errors[steady]
    plot_aligned_velocities_and_errors(aligned_time[steady], aligned_wheel_yaw_rate[steady], aligned_mocap_yaw_rate[steady], filename=output_dir / "steady_aligned_velocities.png")

    print(f"\n Test: {BAG}")
    print(f"\nSteady state yaw rate error statistics (between {motion_start} and {motion_end} seconds):")
    print(f"Mean encoder yaw rate error: {np.mean(steady_yaw_rate_errors):.10g} rad/s")
    print(f"Mean mocap yaw rate: {np.mean(aligned_mocap_yaw_rate[steady]):.10g} rad/s")
    print(f"Steady bias (Mean encoder yaw rate error): {np.mean(steady_yaw_rate_errors):.10g} rad/s")
    print(f"Steady variance (variance of yaw rate error): {np.var(steady_yaw_rate_errors):.10g} (rad/s)^2")
    print(f"Steady standard deviation (standard deviation of yaw rate error): {np.std(steady_yaw_rate_errors):.10g} rad/s")
    print("Encoder/mocap yaw rate ratio: {:.10g}".format(np.mean(aligned_wheel_yaw_rate[steady]) / np.mean(aligned_mocap_yaw_rate[steady])))
    




    

if __name__ == "__main__":
    main()

