import rosbag2_py

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
import numpy as np


bag_path = (
    "rosbag/imu_testing/"
    "rosbag2_2026_09_04-14_00_58_stationary_5min/rosbag2_2026_09_04-14_00_58_0.db3"
)

topic_name = "/mavros_fcu/mavros_fcu/data_raw"

def read_z_values():
    w_values = []

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3",
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    reader.set_filter(
        rosbag2_py.StorageFilter(topics=[topic_name])
    )

    while reader.has_next():
        topic, serialized_data, timestamp = reader.read_next()

        imu_message = deserialize_message(
            serialized_data,
            Imu,
        )

        w_values.append([imu_message.angular_velocity.x, imu_message.angular_velocity.y, imu_message.angular_velocity.z])

    return w_values

def main():
    w_values = read_z_values()

    wz_values = [w[2] for w in w_values]
    wz = np.asarray(wz_values)

    n_wz = len(wz)
    mean_wz = np.mean(wz)
    variance_wz = np.var(wz, ddof=1)
    std_wz = np.std(wz, ddof=1)

    print("\nIMU Angular Velocity Z-axis:")
    print(f"N:                   {n_wz}")
    print(f"Mean/bias:           {mean_wz:.10g} rad/s")
    print(f"Variance:            {variance_wz:.10g} (rad/s)^2")
    print(f"Standard deviation:  {std_wz:.10g} rad/s")


    angular_velocities = np.array(w_values)
    gyro_covariance = np.cov(angular_velocities, rowvar=False, ddof=1)

    print("\nGyroscope Covariance Matrix:")
    print(gyro_covariance)

if __name__ == "__main__":
    main()