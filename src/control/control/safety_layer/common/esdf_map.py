import numpy as np
from rclpy.time import Time

from .constants import NVBLOX_MAX_DISTANCE_M, ODOM_FRAME, PATCH_SIZE
from .frames import transform_pose_2d, yaw_from_quaternion


class EsdfMap:
    """Holds the latest ESDF slice and serves square patches cut from it."""

    def __init__(self, patch_size=PATCH_SIZE):
        self.half_patch = patch_size // 2
        self.vals = None
        self.is_unknown = None
        self.origin = None
        self.resolution = None
        self.frame_id = None

    def store(self, msg):
        """Cache a DistanceMapSlice; unknown cells are replaced by the max distance."""
        vals = np.array(msg.data).reshape(msg.height, msg.width)
        self.is_unknown = (vals == msg.unknown_value)
        self.vals = np.where(self.is_unknown, NVBLOX_MAX_DISTANCE_M, vals)
        self.origin = np.array([msg.origin.x, msg.origin.y])
        self.resolution = msg.resolution
        self.frame_id = msg.header.frame_id

    @staticmethod
    def _clipped_range(desired_start, desired_end, axis_length):
        """Clip a requested [start, end) range to the axis. Returns (slice into the source,
        matching slice into the patch). Both ends are clamped, so a window entirely off the
        axis gives an empty slice rather than wrapping around."""
        valid_start = min(max(desired_start, 0), axis_length)
        valid_end = max(min(desired_end, axis_length), 0)
        source_slice = slice(valid_start, valid_end)
        patch_slice = slice(valid_start - desired_start, valid_end - desired_start)
        return source_slice, patch_slice

    def get_patch(self, cp, source=None, fill_value=None):
        """Square patch cut from `source` (default: the ESDF), centred on `cp` in the map frame.
        Cells outside the ESDF are `fill_value` (default: the max distance)."""
        if source is None:
            source = self.vals
        if fill_value is None:
            fill_value = NVBLOX_MAX_DISTANCE_M

        size = 2 * self.half_patch
        row0 = round((cp[1] - self.origin[1]) / self.resolution)
        col0 = round((cp[0] - self.origin[0]) / self.resolution)

        patch = np.full((size, size), fill_value)

        grid_rows, patch_rows = self._clipped_range(row0 - self.half_patch, row0 + self.half_patch, source.shape[0])
        grid_cols, patch_cols = self._clipped_range(col0 - self.half_patch, col0 + self.half_patch, source.shape[1])

        patch[patch_rows, patch_cols] = source[grid_rows, grid_cols]
        return patch

    def patch_at(self, tf_buffer, x0):
        """(psi, patch) for the robot pose `x0` = (x, y, theta) in the odom frame: the patch
        centred on the robot, and the yaw of the odom frame in the ESDF frame. Returns None
        if no ESDF has arrived yet; raises tf2_ros.TransformException if the odom -> ESDF
        transform is unavailable."""
        if self.vals is None:
            return None
        tf = tf_buffer.lookup_transform(self.frame_id, ODOM_FRAME, Time())
        x_esdf, y_esdf, _ = transform_pose_2d(x0[0], x0[1], x0[2], tf)
        psi = yaw_from_quaternion(tf.transform.rotation)
        return psi, self.get_patch([x_esdf, y_esdf])
