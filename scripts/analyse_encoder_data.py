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

BAG = "mocap_pilot_straight_0.3/t5"
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

def plot_aligned_velocities_and_errors(aligned_time, aligned_wheel_velocity, aligned_mocap_velocity, filename=None):
    plt.figure()
    plt.plot(aligned_time, aligned_wheel_velocity, label="Aligned Wheel Velocity")
    plt.plot(aligned_time, aligned_mocap_velocity, label="Aligned Mocap Velocity")
    plt.title("Aligned Velocities")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    
    if filename:
        plt.savefig(filename)
    plt.show()

def time_sync_velocities(mocap_data, wheel_data):
    # ---------------------------------------------
    # resample position data
    # ---------------------------------------------
    time = mocap_data['timestamps']
    position = np.column_stack([mocap_data['positions']['x'], mocap_data['positions']['y'], mocap_data['positions']['z']])

    dt = np.mean(np.diff(time))
    uniform_time = np.arange(time[0], time[-1], dt)

    uniform_position = np.column_stack([
        np.interp(uniform_time, time, position[:,0]),
        np.interp(uniform_time, time, position[:,1]),
        np.interp(uniform_time, time, position[:,2])
    ])

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


    mocap_velocities_smooth = savgol_filter(
        uniform_position,
        window_length,
        polyorder,
        deriv=1,
        delta=dt,
        axis=0
    )

    mocap_resultant_velocity = np.linalg.norm(mocap_velocities_smooth[:,:2], axis=1)
    wheel_resultant_velocity = np.linalg.norm([
        wheel_data['velocities']['linear']['x'],
        wheel_data['velocities']['linear']['y']
    ], axis=0)

    wheel_time = wheel_data["timestamps"]

    start = max(uniform_time[0], wheel_time[0])
    end = min(uniform_time[-1], wheel_time[-1])
    t_common = np.arange(start, end, dt)

    mocap_interp = np.interp(
        t_common, uniform_time, mocap_resultant_velocity
    )
    wheel_interp = np.interp(
        t_common, wheel_time, wheel_resultant_velocity
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
    wheel_speed = wheel_resultant_velocity[1:]

    query_time = comparison_time + time_delay

    valid = (
        (query_time >= uniform_time[0])
        & (query_time <= uniform_time[-1])
    )

    aligned_time = comparison_time[valid]
    aligned_wheel_velocity = wheel_speed[valid]
    aligned_mocap_velocity = np.interp(
        query_time[valid],
        uniform_time,
        mocap_resultant_velocity,
    )

    return (
        aligned_time,
        aligned_wheel_velocity,
        aligned_mocap_velocity,
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


    aligned_time, aligned_wheel_velocity, aligned_mocap_velocity = (time_sync_velocities(mocap_data, wheel_data))

    plot_aligned_velocities_and_errors(aligned_time, aligned_wheel_velocity, aligned_mocap_velocity, filename=output_dir / "aligned_velocities.png")


    # Calculate covariance of the velocity errors to estimate the R covariance matrix entry for velocity
    velocity_errors = aligned_wheel_velocity - aligned_mocap_velocity
    R_velocity = np.var(velocity_errors)    


    print(f"\nEKF R covariance matrix entry for velocity: {R_velocity:.10g} (m/s)^2")
    print(f"Mean velocity error: {np.mean(velocity_errors):.10g} m/s")
    print(f"Variance of velocity error: {np.var(velocity_errors):.10g} (m/s)^2")
    print(f"Standard deviation of velocity error: {np.std(velocity_errors):.10g} m/s")
    print(f"Max velocity error: {np.max(velocity_errors):.10g} m/s")
    print(f"Min velocity error: {np.min(velocity_errors):.10g} m/s")

    motion_start = 32.0
    motion_end = 38.0

    steady = (aligned_time >= motion_start) & (aligned_time <= motion_end)
    steady_velocity_errors = velocity_errors[steady]
    plot_aligned_velocities_and_errors(aligned_time[steady], aligned_wheel_velocity[steady], aligned_mocap_velocity[steady], filename=output_dir / "steady_aligned_velocities.png")

    print(f"\n Test: {BAG}")
    print(f"\nSteady state velocity error statistics (between {motion_start} and {motion_end} seconds):")
    print(f"Mean encoder speed error: {np.mean(steady_velocity_errors):.10g} m/s")
    print(f"Mean mocap speed: {np.mean(aligned_mocap_velocity[steady]):.10g} m/s")
    print(f"Steady bias (Mean encoder speed error): {np.mean(steady_velocity_errors):.10g} m/s")
    print(f"Steady variance (variance of speed error): {np.var(steady_velocity_errors):.10g} (m/s)^2")
    print(f"Steady standard deviation (standard deviation of speed error): {np.std(steady_velocity_errors):.10g} m/s")
    print("Encoder/mocap speed ratio: {:.10g}".format(np.mean(aligned_wheel_velocity[steady]) / np.mean(aligned_mocap_velocity[steady])))
    




    

if __name__ == "__main__":
    main()

