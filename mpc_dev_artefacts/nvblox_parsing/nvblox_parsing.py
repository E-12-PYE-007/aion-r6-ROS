# Confirm ability to parse NVblox messages and extract coefficients for knots

import sqlite3
import numpy as np

from rclpy.serialization import deserialize_message
from nvblox_msgs.msg import DistanceMapSlice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline

NVBLOX_MAX_DISTANCE_M = 2.0 # Max distance in ESDF representation. Should match esdf config.
PATCH_SIZE = 22 # Number of grid points to select. At 0.1m spacing, gives a 2.1m grid
HALF_PATCH_SIZE = PATCH_SIZE//2

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
        axis. Returns (slice into the source array, matching slice into the patch)."""
        valid_start = max(desired_start, 0)
        valid_end = min(desired_end, axis_length)
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

# Open sqlite connection 
BAG = "src/nvblox_MPC_extras/rosbags/esdf_single_obs/esdf_single_obs_0.db3"

con = sqlite3.connect(BAG)
cur = con.cursor()

# Query the rosbags to get the topic ID for static_map_slice

topics = cur.execute("SELECT id, name, type FROM topics").fetchall()
for t in topics:
    print(t)

topic_id = [t[0] for t in topics if t[1] == "/nvblox_node/static_map_slice"][0]

# Get the messages sent under the static_map_slice topic
rows = cur.execute(
    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (topic_id,)
).fetchall()

print(f"{len(rows)} messages found")
print(f"first message is {len(rows[0][0])} bytes")

# Deserialise the first message

msg = deserialize_message(rows[0][0], DistanceMapSlice)

print("frame:", msg.header.frame_id)
print("resolution:", msg.resolution)
print("width:", msg.width, "height:", msg.height)
print("origin:", msg.origin.x, msg.origin.y, msg.origin.z)
print("unknown_value:", msg.unknown_value)
print("data length:", len(msg.data))

# Reshape the flat array to a 2D array (more intuitive to work with)
esdf_map = EsdfMap()
esdf_map.store(msg)

# Plot for visualisation and confirmation that reshaping is correct
print(f"Resized to {esdf_map.vals.shape}")
plt.imshow(esdf_map.vals, origin='lower')
plt.colorbar()
plt.savefig('grid_check.png')
plt.close()

# Visualise the interpolation process: discretised cells, a continuous surface,
# and a top-down 2D view, all built from one patch (the whole grid was too much
# to read at once). Sentinel cells (originally "unknown", plus anything padded
# in from outside the grid) are set to null rather than a fake distance value,
# since a fabricated flat value is exactly what was producing the sharp, unreal
# jumps in the fit.
# Found by searching for a fully-known patch with a small obstacle notch
# concentrated in one corner (rather than straddling the centre), so the shape
# reads clearly without one sign overwhelming the plot.
centre_col, centre_row = 81, 46  # voxels
centre_x = esdf_map.origin[0] + centre_col * esdf_map.resolution
centre_y = esdf_map.origin[1] + centre_row * esdf_map.resolution

vals_for_display = np.where(esdf_map.is_unknown, np.nan, esdf_map.vals)
patch = esdf_map.get_patch([centre_x, centre_y], source=vals_for_display, fill_value=np.nan)
n_rows, n_cols = patch.shape

# Cell-corner coordinates in metres, local to the patch
row_coords = np.arange(n_rows) * esdf_map.resolution
col_coords = np.arange(n_cols) * esdf_map.resolution

# ESDF values are signed (negative = inside an obstacle, positive = free space), so
# zero is a meaningful midpoint, not an arbitrary one - a diverging colour map
# centred on it reads far better than a plain sequential one like viridis. Built
# from scratch (rather than e.g. RdBu) because the stock diverging maps fade to
# white at the centre, which is invisible against a white plot background - a
# mid grey midpoint stays visible everywhere without looking dark overall.
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('esdf_diverging', ['#0072B2', '#999999', '#27AE60'])
cmap.set_bad(alpha=0)  # sentinel (NaN) cells render as nothing
# Scaled to this patch's own data, not the ESDF's full +-2m physical range - this
# patch's real variation only spans about +-0.5m (it's a smooth field close to a
# boundary, so it's small there by nature), and fixing the scale to +-2m squashed
# all of that real structure into a thin grey band around the centre. Percentiles
# (not min/max) so the single stray -2.0 noise spike found earlier doesn't blow
# the scale back out - a fixed scale is the right call when comparing multiple
# patches to each other, but for reading the shape of one patch, this is better.
known_vals = patch[~np.isnan(patch)]
vmin = np.percentile(known_vals, 2)
vmax = np.percentile(known_vals, 98)
norm = TwoSlopeNorm(vmin=min(vmin, -1e-3), vcenter=0.0, vmax=max(vmax, 1e-3))

fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(wspace=0.4)

BLANK_BG = '#e6e6e6'  # distinct from the colour map so blank (sentinel) cells are obvious

# Picked by comparing several angles side by side - this one (matplotlib's own
# long-standing default, as it happens) gave the clearest read of the terrain
# relief, the zero-crossing band, and the obstacle spike without them overlapping.
CAMERA_ELEV = 30
CAMERA_AZIM = -60

def _set_3d_background(ax):
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(BLANK_BG)

# Left: one square column per cell, height equal to the cell's distance value.
# Sentinel cells are dropped entirely rather than drawn as a fake column.
ax_discrete = fig.add_subplot(1, 3, 1, projection='3d')
col_grid, row_grid = np.meshgrid(col_coords, row_coords)
col_flat, row_flat, patch_flat = col_grid.ravel(), row_grid.ravel(), patch.ravel()
known = ~np.isnan(patch_flat)
ax_discrete.bar3d(
    col_flat[known], row_flat[known], np.zeros(known.sum()),
    esdf_map.resolution, esdf_map.resolution, patch_flat[known],
    color=cmap(norm(patch_flat[known])), edgecolor='#333333', linewidth=0.3, shade=True
)
ax_discrete.set_zlim(-NVBLOX_MAX_DISTANCE_M, NVBLOX_MAX_DISTANCE_M)  # bar3d autoscale ignores negative dz with per-bar colours
ax_discrete.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
_set_3d_background(ax_discrete)
ax_discrete.set_title("Discretised ESDF")
ax_discrete.set_xlabel("x (m)")
ax_discrete.set_ylabel("y (m)")
ax_discrete.set_zlabel("distance (m)")

# Middle: bicubic spline fit through the same cells, evaluated on a fine grid. The
# spline itself needs real numbers (no NaNs), so it's fit on the solver-realistic
# patch (sentinels clamped, same as what the solver actually sees) - but any fine
# point whose nearest original cell was a sentinel is then nulled out afterwards,
# so the fit's own artefacts near those cells don't get displayed as if real.
patch_for_fit = esdf_map.get_patch([centre_x, centre_y])
sentinel_patch = esdf_map.get_patch(
    [centre_x, centre_y], source=esdf_map.is_unknown.astype(float), fill_value=1.0
) > 0.5

spline = RectBivariateSpline(row_coords, col_coords, patch_for_fit, kx=3, ky=3)
n_fine = max(200, n_rows * 4)
row_fine = np.linspace(row_coords[0], row_coords[-1], n_fine)
col_fine = np.linspace(col_coords[0], col_coords[-1], n_fine)
surface = spline(row_fine, col_fine)

nearest_row = np.clip(np.round(row_fine / esdf_map.resolution).astype(int), 0, n_rows - 1)
nearest_col = np.clip(np.round(col_fine / esdf_map.resolution).astype(int), 0, n_cols - 1)
surface[sentinel_patch[np.ix_(nearest_row, nearest_col)]] = np.nan

ax_surface = fig.add_subplot(1, 3, 2, projection='3d')
col_fine_grid, row_fine_grid = np.meshgrid(col_fine, row_fine)
ax_surface.plot_surface(col_fine_grid, row_fine_grid, surface, cmap=cmap, norm=norm, shade=True)
ax_surface.view_init(elev=CAMERA_ELEV, azim=CAMERA_AZIM)
_set_3d_background(ax_surface)
ax_surface.set_title("Continuous bicubic interpolation")
ax_surface.set_xlabel("x (m)")
ax_surface.set_ylabel("y (m)")
ax_surface.set_zlabel("distance (m)")

# Right: top-down 2D view of the same patch, sentinels blank, same colour scale.
ax_2d = fig.add_subplot(1, 3, 3)
im = ax_2d.imshow(
    patch, origin='lower', cmap=cmap, norm=norm,
    extent=[col_coords[0], col_coords[-1], row_coords[0], row_coords[-1]]
)
ax_2d.set_facecolor(BLANK_BG)
ax_2d.set_title("Top-down view")
ax_2d.set_xlabel("x (m)")
ax_2d.set_ylabel("y (m)")
fig.colorbar(im, ax=ax_2d, label="distance (m)", shrink=0.8)

plt.savefig('interpolation_comparison.png', bbox_inches='tight')
plt.close()
