import numpy as np

from .constants import NVBLOX_MAX_DISTANCE_M, PATCH_SIZE

HALF_PATCH_SIZE = PATCH_SIZE // 2


class EsdfMap:
    """Holds the latest ESDF slice and serves patches cut from it."""

    def __init__(self):
        self.vals = None
        self.is_unknown = None
        self.origin = None
        self.resolution = None
        self.frame_id = None

    def store(self, msg):
        """Deserialised-message callback: reshape, clean, and cache the slice."""
        vals = np.array(msg.data).reshape(msg.height, msg.width)
        self.is_unknown = (vals == msg.unknown_value)
        self.vals = np.where(self.is_unknown, NVBLOX_MAX_DISTANCE_M, vals)
        self.origin = np.array([msg.origin.x, msg.origin.y])
        self.resolution = msg.resolution
        self.frame_id = msg.header.frame_id

    @staticmethod
    def _clipped_range(desired_start, desired_end, axis_length):
        """Clip a requested [start, end) range to what actually exists along one
        axis. Returns (slice into the source array, matching slice into the patch).

        Both bounds are clamped into [0, axis_length] - clamping only valid_start
        (via max(...,0)) and valid_end (via min(...,axis_length)) individually isn't
        enough: if the whole requested window lies off the left edge, valid_end can
        itself come out negative, and slice(0, negative) doesn't mean "empty" - it's
        reinterpreted as counting from the array's own end, silently pulling in real
        data from the wrong place instead of clipping to nothing."""
        valid_start = min(max(desired_start, 0), axis_length)
        valid_end = max(min(desired_end, axis_length), 0)
        source_slice = slice(valid_start, valid_end)
        patch_slice = slice(valid_start - desired_start, valid_end - desired_start)
        return source_slice, patch_slice

    def get_patch(self, cp, source=None, fill_value=None):
        """ Returns a square patch of PATCH_SIZE cut from `source` (defaults to self.vals), centred on cp
        in the map frame and clipped to `fill_value` (defaults to the max-distance clamp) if out of bounds.
        The default call is what's fed to the solver. Queries must be made in map frame (i.e. solver needs
        to rotate X_k expressed in odom)"""
        if source is None:
            source = self.vals
        if fill_value is None:
            fill_value = NVBLOX_MAX_DISTANCE_M

        size = 2 * HALF_PATCH_SIZE
        row0 = round((cp[1] - self.origin[1]) / self.resolution)
        col0 = round((cp[0] - self.origin[0]) / self.resolution)

        patch = np.full((size, size), fill_value)

        grid_rows, patch_rows = self._clipped_range(row0 - HALF_PATCH_SIZE, row0 + HALF_PATCH_SIZE, source.shape[0])
        grid_cols, patch_cols = self._clipped_range(col0 - HALF_PATCH_SIZE, col0 + HALF_PATCH_SIZE, source.shape[1])

        patch[patch_rows, patch_cols] = source[grid_rows, grid_cols]
        return patch
