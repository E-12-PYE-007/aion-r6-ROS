#!/usr/bin/env python3
"""Plots and animates a sim_robot CSV log. Standalone: needs numpy and matplotlib (plus
ffmpeg for mp4), no ROS.

  python3 safety_layer_bench_testing/plot_run.py                  # the newest log in sim_logs/
  python3 safety_layer_bench_testing/plot_run.py LOG.csv          # a specific log
  python3 safety_layer_bench_testing/plot_run.py --show
  python3 safety_layer_bench_testing/plot_run.py --png final.png

The ESDF is read from the rosbag named in the log's header. Cells with negative distance
(inside an obstacle) are left empty; cells with non-negative distance use a
red (at an obstacle surface) -> yellow -> green -> cyan -> blue -> purple (2 m away)
gradient, like a rainbow-coloured ESDF in RViz; unknown cells are light grey.

Each frame shows the robot's path so far, its pose, and the action chunk it was last given,
drawn in the odom frame anchored at the robot's pose when that chunk (seq_num) first appeared
in the log - the same anchoring the controllers use. Below the map: v and omega, the safety
layer's slack, and the ESDF distance at the robot's centre, with a cursor at the frame's time.
"""
import argparse
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import time

import numpy as np

ESDF_TOPIC = '/nvblox_node/static_map_slice'
NVBLOX_MAX_DISTANCE_M = 2.0     # unknown cells are treated as this distance, as the controllers do
N_WAYPOINTS = 8
LOG_COLUMNS_BEFORE_CHUNK = 8    # t, x, y, theta, v, omega, slack, seq_num

# Left = at an obstacle surface (ESDF 0), right = NVBLOX_MAX_DISTANCE_M.
ESDF_COLORS = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#a020f0']
UNKNOWN_COLOR = (0.90, 0.90, 0.90, 1.0)
PATH_COLOR = '#000000'
CHUNK_COLOR = '#e6007e'
V_COLOR = '#0057b8'
OMEGA_COLOR = '#d95f02'

SLACK_LABELS = {'cbf': 'slack [m/s]', 'mpc': 'slack [m]'}


# ---------------------------------------------------------------- ESDF from the rosbag

class _CdrReader:
    """Little-endian CDR reader. Alignment is relative to the byte after the 4-byte header."""

    def __init__(self, buf):
        if len(buf) < 4 or buf[1] & 1 != 1:
            raise ValueError('only little-endian CDR messages are supported')
        self.buf = buf
        self.pos = 4

    def read(self, fmt):
        size = struct.calcsize(fmt)
        self.pos += (-(self.pos - 4)) % size
        (value,) = struct.unpack_from('<' + fmt, self.buf, self.pos)
        self.pos += size
        return value

    def read_string(self):
        length = self.read('I')
        raw = self.buf[self.pos:self.pos + length]
        self.pos += length
        return raw.rstrip(b'\x00').decode()

    def read_float_array(self):
        count = self.read('I')
        self.pos += (-(self.pos - 4)) % 4
        values = np.frombuffer(self.buf, dtype='<f4', count=count, offset=self.pos)
        self.pos += 4 * count
        return values


def parse_distance_map_slice(buf):
    """nvblox_msgs/DistanceMapSlice: header, resolution, width, height, origin (Point),
    unknown_value, data[]."""
    r = _CdrReader(buf)
    r.read('i')                  # header.stamp.sec
    r.read('I')                  # header.stamp.nanosec
    frame_id = r.read_string()
    resolution = r.read('f')
    width = r.read('I')
    height = r.read('I')
    origin_x = r.read('d')
    origin_y = r.read('d')
    r.read('d')                  # origin.z
    unknown_value = r.read('f')
    data = r.read_float_array()
    return dict(frame_id=frame_id, resolution=resolution, width=width, height=height,
                origin=(origin_x, origin_y), unknown_value=unknown_value, data=data)


class Esdf:
    """The first ESDF slice in a rosbag. values[row, col]: distance at
    (origin_x + col * resolution, origin_y + row * resolution); unknown cells are NaN."""

    def __init__(self, bag_path):
        con = sqlite3.connect(bag_path)
        try:
            cur = con.cursor()
            topic_ids = [row[0] for row in cur.execute('SELECT id, name FROM topics')
                         if row[1] == ESDF_TOPIC]
            if not topic_ids:
                raise ValueError(f'{bag_path} has no {ESDF_TOPIC} topic')
            row = cur.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1',
                              (topic_ids[0],)).fetchone()
        finally:
            con.close()
        if row is None:
            raise ValueError(f'{bag_path} has no messages on {ESDF_TOPIC}')

        msg = parse_distance_map_slice(bytes(row[0]))
        raw = msg['data'].reshape(msg['height'], msg['width']).astype(float)
        self.frame_id = msg['frame_id']
        self.resolution = msg['resolution']
        self.origin = msg['origin']
        self.unknown = raw == msg['unknown_value']
        self.values = np.where(self.unknown, np.nan, raw)
        ys, xs = np.mgrid[0:msg['height'], 0:msg['width']]
        self.grid_x = self.origin[0] + xs * self.resolution
        self.grid_y = self.origin[1] + ys * self.resolution

    @property
    def extent(self):
        """imshow extent (left, right, bottom, top): origin is the centre of cell (0, 0)."""
        half = self.resolution / 2.0
        return (self.grid_x[0, 0] - half, self.grid_x[0, -1] + half,
                self.grid_y[0, 0] - half, self.grid_y[-1, 0] + half)

    def distance_at(self, x, y):
        """Nearest-cell distance at points (arrays); unknown cells read as the max distance,
        as in the controllers. Points off the grid are NaN."""
        col = np.rint((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        row = np.rint((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        inside = (row >= 0) & (row < self.values.shape[0]) & (col >= 0) & (col < self.values.shape[1])
        out = np.full(np.shape(x), np.nan)
        looked_up = self.values[row[inside], col[inside]]
        out[inside] = np.where(np.isnan(looked_up), NVBLOX_MAX_DISTANCE_M, looked_up)
        return out


# ---------------------------------------------------------------- the log

class Run:
    def __init__(self, path):
        header = {}
        rows = []
        with open(path) as f:
            lines = f.read().splitlines()
        i = 0
        while i < len(lines) and lines[i].startswith('#'):
            key, _, value = lines[i][1:].partition(':')
            header[key.strip()] = value.strip()
            i += 1
        columns = lines[i].split(',')
        for line in lines[i + 1:]:
            fields = line.split(',')
            if len(fields) == len(columns):    # a run killed mid-write can leave a partial last row
                rows.append([float(value) for value in fields])
        if not rows:
            raise ValueError(f'{path} has a header but no data rows')
        data = np.array(rows)

        self.header = header
        self.controller = header.get('controller', 'unknown')
        self.esdf_bag = header['esdf_bag']
        self.start = np.array([float(v) for v in header['start'].split()])
        self.goal_distance = float(header['goal_distance'])
        self.t, self.x, self.y, self.theta = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        self.v, self.omega, self.slack = data[:, 4], data[:, 5], data[:, 6]
        self.seq = data[:, 7].astype(int)
        self.chunk = data[:, LOG_COLUMNS_BEFORE_CHUNK:].reshape(len(data), N_WAYPOINTS, 3)

        self.goal = self.start[:2] + self.goal_distance * np.array(
            [math.cos(self.start[2]), math.sin(self.start[2])])

        self.first_row_of_seq = {}
        for row_index, seq in enumerate(self.seq):
            if seq >= 0 and seq not in self.first_row_of_seq:
                self.first_row_of_seq[seq] = row_index

    def chunk_in_odom(self, row_index):
        """(9, 2) array: the anchor pose then the 8 chunk waypoints, in the odom frame; None
        before the first chunk."""
        seq = self.seq[row_index]
        if seq < 0:
            return None
        a = self.first_row_of_seq[seq]
        ax, ay, ath = self.x[a], self.y[a], self.theta[a]
        rel = self.chunk[row_index]
        c, s = math.cos(ath), math.sin(ath)
        wx = ax + rel[:, 0] * c - rel[:, 1] * s
        wy = ay + rel[:, 0] * s + rel[:, 1] * c
        return np.vstack([[ax, ay], np.stack([wx, wy], axis=1)])


# ---------------------------------------------------------------- drawing

def robot_triangle(x, y, theta, size):
    pts = np.array([[size, 0.0], [-0.6 * size, 0.5 * size], [-0.6 * size, -0.5 * size]])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return pts @ rot.T + np.array([x, y])


def default_view(run, min_x_span=4.0, min_y_span=3.0, pad=0.8):
    xs = np.concatenate([run.x, [run.start[0], run.goal[0]]])
    ys = np.concatenate([run.y, [run.start[1], run.goal[1]]])
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    if x1 - x0 < min_x_span:
        mid = (x0 + x1) / 2.0
        x0, x1 = mid - min_x_span / 2.0, mid + min_x_span / 2.0
    if y1 - y0 < min_y_span:
        mid = (y0 + y1) / 2.0
        y0, y1 = mid - min_y_span / 2.0, mid + min_y_span / 2.0
    return x0, x1, y0, y1


class RunFigure:
    def __init__(self, plt, run, esdf, margin, robot_size, view, dpi, title=None):
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Polygon

        self.run, self.esdf = run, esdf
        self.distance_at_robot = esdf.distance_at(run.x, run.y)

        self.fig = plt.figure(figsize=(12, 9), dpi=dpi)
        if title:
            self.fig.suptitle(title, fontsize=17, fontweight='bold', y=0.985)
        grid = self.fig.add_gridspec(4, 1, height_ratios=[6, 1.2, 1.2, 1.2], hspace=0.35)
        self.ax = self.fig.add_subplot(grid[0])
        strip_axes = [self.fig.add_subplot(grid[i]) for i in (1, 2, 3)]
        self.ax_v, self.ax_slack, self.ax_dist = strip_axes

        # --- map
        ax = self.ax
        cmap = LinearSegmentedColormap.from_list('esdf_rainbow', ESDF_COLORS)
        unknown_layer = np.zeros(esdf.unknown.shape + (4,))
        unknown_layer[esdf.unknown] = UNKNOWN_COLOR
        ax.imshow(unknown_layer, origin='lower', extent=esdf.extent, interpolation='nearest', zorder=1)
        shown = np.ma.masked_where(np.isnan(esdf.values) | (esdf.values < 0.0), esdf.values)
        image = ax.imshow(shown, origin='lower', extent=esdf.extent, cmap=cmap, vmin=0.0,
                          vmax=NVBLOX_MAX_DISTANCE_M, interpolation='nearest', zorder=2)
        ax.contour(esdf.grid_x, esdf.grid_y, esdf.values, levels=[0.0], colors='k',
                   linewidths=1.0, zorder=3)
        ax.contour(esdf.grid_x, esdf.grid_y, esdf.values, levels=[margin], colors='k',
                   linewidths=0.8, linestyles='dashed', zorder=3)
        cax = ax.inset_axes([1.01, 0.0, 0.02, 1.0])
        colorbar = self.fig.colorbar(image, cax=cax)
        colorbar.set_label('ESDF distance [m]  (negative = inside obstacle: empty)')

        ax.plot([run.start[0], run.goal[0]], [run.start[1], run.goal[1]], color='0.35',
                linestyle=':', linewidth=1.2, zorder=4)
        ax.plot(*run.start[:2], 'o', color='k', markersize=7, zorder=6, label='start')
        ax.plot(*run.goal, '*', color='k', markersize=13, zorder=6, label='goal')
        ax.plot(run.x, run.y, color=PATH_COLOR, alpha=0.18, linewidth=1.5, zorder=4)
        (self.trail,) = ax.plot([], [], color=PATH_COLOR, linewidth=2.2, zorder=5, label='path')
        (self.chunk_line,) = ax.plot([], [], '-o', color=CHUNK_COLOR, linewidth=1.6, markersize=4,
                                     zorder=7, label='action chunk')
        self.robot = Polygon(robot_triangle(run.x[0], run.y[0], run.theta[0], robot_size),
                             closed=True, facecolor='white', edgecolor='k', linewidth=1.8, zorder=8)
        ax.add_patch(self.robot)
        self.robot_size = robot_size
        self.text = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
                            family='monospace', fontsize=10, zorder=9,
                            bbox=dict(facecolor='white', edgecolor='0.6', alpha=0.9))
        ax.set_aspect('equal', adjustable='datalim')
        x0, x1, y0, y1 = view
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel('x [m] (odom)')
        ax.set_ylabel('y [m] (odom)')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_title(f'{run.controller}   start ({run.start[0]:.2f}, {run.start[1]:.2f}, '
                     f'{math.degrees(run.start[2]):.0f}°)   margin {margin:.2f} m (dashed)')

        # --- strips
        t_end = run.t[-1]
        self.ax_v.plot(run.t, run.v, color=V_COLOR, label='v [m/s]')
        self.ax_v.plot(run.t, run.omega, color=OMEGA_COLOR, label='ω [rad/s]')
        self.ax_v.set_ylabel('cmd_vel')
        self.ax_v.legend(loc='upper right', ncol=2, fontsize=8)
        self.ax_slack.plot(run.t, run.slack, color='k')
        self.ax_slack.set_ylabel(SLACK_LABELS.get(run.controller, 'slack'))
        self.ax_dist.plot(run.t, self.distance_at_robot, color='k')
        self.ax_dist.axhline(margin, color='k', linestyle='--', linewidth=0.8)
        self.ax_dist.set_ylabel('ESDF at\nrobot [m]')
        self.ax_dist.set_xlabel('t [s]')
        self.cursors = []
        for strip in strip_axes:
            strip.set_xlim(0.0, max(t_end, 1e-6))
            strip.grid(alpha=0.3)
            self.cursors.append(strip.axvline(0.0, color=CHUNK_COLOR, linewidth=1.5))
        for strip in (self.ax_v, self.ax_slack):
            strip.tick_params(labelbottom=False)

        # Everything that changes from frame to frame, in draw order. The rest of the figure
        # is drawn once and reused (see capture_background).
        self.dynamic = [(self.ax, self.trail), (self.ax, self.chunk_line), (self.ax, self.robot),
                        (self.ax, self.text)] + list(zip(strip_axes, self.cursors))
        self.background = None

    def capture_background(self):
        """Draw everything except the dynamic artists once, and keep the pixels."""
        for _, artist in self.dynamic:
            artist.set_animated(True)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def frame_rgba(self, i):
        """Frame i as raw RGBA pixels: the saved background with only the dynamic artists redrawn.
        The returned buffer is reused by the next call."""
        canvas = self.fig.canvas
        self.draw_row(i)
        canvas.restore_region(self.background)
        for ax, artist in self.dynamic:
            ax.draw_artist(artist)
        return canvas.buffer_rgba()

    def draw_row(self, i):
        run = self.run
        self.trail.set_data(run.x[:i + 1], run.y[:i + 1])
        self.robot.set_xy(robot_triangle(run.x[i], run.y[i], run.theta[i], self.robot_size))
        chunk = run.chunk_in_odom(i)
        if chunk is None:
            self.chunk_line.set_data([], [])
        else:
            self.chunk_line.set_data(chunk[:, 0], chunk[:, 1])
        for cursor in self.cursors:
            cursor.set_xdata([run.t[i], run.t[i]])
        dist = self.distance_at_robot[i]
        dist_text = '  n/a ' if np.isnan(dist) else f'{dist:6.3f}'
        self.text.set_text(
            f't = {run.t[i]:6.2f} s   pose = ({run.x[i]:5.2f}, {run.y[i]:5.2f}, '
            f'{math.degrees(run.theta[i]):6.1f}°)\n'
            f'v = {run.v[i]:5.3f}   ω = {run.omega[i]:6.3f}   slack = {run.slack[i]:6.3f}\n'
            f'ESDF at robot = {dist_text} m   chunk seq = {run.seq[i]}')
        return [artist for _, artist in self.dynamic]


def frame_rows(run, fps, speed):
    """Log rows to show, one per video frame: the row nearest each frame's time, so playback
    runs at `speed` times real time whatever the log's own rate. Always ends on the last row."""
    frame_times = np.arange(0.0, run.t[-1], speed / fps)
    rows = np.searchsorted(run.t, frame_times + run.t[0])
    rows = np.minimum(rows, len(run.t) - 1)
    rows = list(dict.fromkeys(int(r) for r in rows))
    if rows[-1] != len(run.t) - 1:
        rows.append(len(run.t) - 1)
    return rows


def write_mp4(figure, rows, out, fps):
    """Pipe raw frames to ffmpeg (H.264). Frame size is the figure's pixel size, rounded down to even."""
    if shutil.which('ffmpeg') is None:
        sys.exit('ffmpeg not found: install it, or use --out RUN.gif, --png, or --show')
    width, height = figure.fig.canvas.get_width_height()
    command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba',
               '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-c:v', 'libx264', '-preset', 'veryfast',
               '-crf', '23', '-pix_fmt', 'yuv420p', out]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for n, row in enumerate(rows, start=1):
            process.stdin.write(figure.frame_rgba(row))
            print_progress(n, len(rows))
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        code = process.wait()
    if code != 0:
        sys.exit(f'\nffmpeg failed (exit code {code})')


def write_gif(figure, rows, out, fps):
    from PIL import Image
    width, height = figure.fig.canvas.get_width_height()
    frames = []
    for n, row in enumerate(rows, start=1):
        image = Image.frombuffer('RGBA', (width, height), bytes(figure.frame_rgba(row)), 'raw', 'RGBA', 0, 1)
        frames.append(image.convert('P', palette=Image.ADAPTIVE))
        print_progress(n, len(rows))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(round(1000.0 / fps)), loop=0)


def print_progress(n, total):
    if n % 50 == 0 or n == total:
        print(f'\r  frame {n}/{total}', end='', flush=True)


def newest_log():
    """The most recently modified CSV in ./sim_logs, else in <repo root>/sim_logs (the directory
    above this script), so it works from any working directory."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    searched = []
    for directory in (os.path.join(os.getcwd(), 'sim_logs'), os.path.join(repo_root, 'sim_logs')):
        if directory in searched:
            continue
        searched.append(directory)
        logs = [os.path.join(directory, name) for name in os.listdir(directory)
                if name.endswith('.csv')] if os.path.isdir(directory) else []
        if logs:
            return max(logs, key=os.path.getmtime)
    sys.exit('no log given and no .csv found in: ' + ', '.join(searched))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log', nargs='?', help='CSV written by sim_robot (default: the newest in sim_logs/)')
    parser.add_argument('--out', help='output video (.mp4 or .gif); default: next to the log, same name')
    parser.add_argument('--png', metavar='PATH', help='save only the final frame as a PNG, no video')
    parser.add_argument('--show', action='store_true', help='play in a window instead of writing a video')
    parser.add_argument('--fps', type=float, default=20.0, help='video frame rate (default 20)')
    parser.add_argument('--speed', type=float, default=1.0, help='playback speed vs. real time (default 1)')
    parser.add_argument('--dpi', type=int, default=100, help='render resolution (default 100 -> 1200x900)')
    parser.add_argument('--title', help='heading shown above the plot')
    parser.add_argument('--margin', type=float, default=0.20, help='safety margin to draw [m] (default 0.20)')
    parser.add_argument('--robot-size', type=float, default=0.12, help='length of the robot marker [m]')
    parser.add_argument('--bag', help='ESDF rosbag; default: the one named in the log header')
    parser.add_argument('--view', type=float, nargs=4, metavar=('X0', 'X1', 'Y0', 'Y1'),
                        help='map view limits in odom [m]; default: fit the run')
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    if args.log is None:
        args.log = newest_log()
        print(f'using the newest log: {args.log}')
    run = Run(args.log)
    esdf = Esdf(args.bag or run.esdf_bag)
    if esdf.frame_id != 'odom':
        print(f'warning: the ESDF frame is {esdf.frame_id!r}, not odom; it is drawn as if they were the same',
              file=sys.stderr)

    view = tuple(args.view) if args.view else default_view(run)
    figure = RunFigure(plt, run, esdf, args.margin, args.robot_size, view, args.dpi, args.title)

    valid = ~np.isnan(figure.distance_at_robot)
    if valid.any():
        k = int(np.nanargmin(figure.distance_at_robot))
        print(f'{len(run.t)} rows, {run.t[-1]:.1f} s; min ESDF at robot centre '
              f'{figure.distance_at_robot[k]:.3f} m at t = {run.t[k]:.2f} s '
              f'(margin {args.margin:.2f} m)')

    if args.png:
        figure.draw_row(len(run.t) - 1)
        figure.fig.savefig(args.png, dpi=args.dpi)
        print(f'wrote {args.png}')
        return

    rows = frame_rows(run, args.fps, args.speed)
    if args.show:
        for _, artist in figure.dynamic:
            artist.set_animated(True)
        anim = animation.FuncAnimation(figure.fig, lambda i: figure.draw_row(rows[i]), frames=len(rows),
                                       interval=1000.0 / args.fps, repeat=False, blit=True)
        plt.show()
        return

    out = args.out or os.path.splitext(args.log)[0] + '.mp4'
    print(f'rendering {len(rows)} frames to {out}')
    figure.capture_background()
    start_time = time.time()
    (write_gif if out.endswith('.gif') else write_mp4)(figure, rows, out, args.fps)
    print(f'\nwrote {out} in {time.time() - start_time:.1f} s')


if __name__ == '__main__':
    main()
