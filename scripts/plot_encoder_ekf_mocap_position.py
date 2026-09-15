#!/usr/bin/env python3
"""Plot recorded mocap, wheel, and EKF positions. No EKF is reconstructed.

Raw X/Y coordinates are in each source's recorded frame. Horizontal displacement
from the common comparison start is invariant to horizontal frame rotation; it
is distance from that starting point, not accumulated path length.
"""
import argparse
import sqlite3
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BAG = REPO / 'rosbag/mocap_pilot_straight_0.3/t1'
TOPICS = {
    '/mocap/rover/pose': ('Mocap', PoseStamped, 'geometry_msgs/msg/PoseStamped'),
    '/odometry/wheel': ('Encoder alone', Odometry, 'nav_msgs/msg/Odometry'),
    '/odometry/filtered': ('EKF', Odometry, 'nav_msgs/msg/Odometry'),
}


def read_positions(bag):
    rows = {topic: [] for topic in TOPICS}
    origin = None
    for database in sorted(bag.glob('*.db3')):
        with sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True) as connection:
            first = connection.execute('SELECT MIN(timestamp) FROM messages').fetchone()[0]
            if first is not None:
                origin = first if origin is None else min(origin, first)
            topics = {i: (name, kind) for i, name, kind in connection.execute('SELECT id,name,type FROM topics')}
            for topic_id, timestamp, data in connection.execute('SELECT topic_id,timestamp,data FROM messages ORDER BY timestamp'):
                topic, kind = topics[topic_id]
                if topic not in TOPICS:
                    continue
                _, message_class, expected = TOPICS[topic]
                if kind != expected:
                    raise ValueError(f'{topic}: expected {expected}, got {kind}')
                msg = deserialize_message(data, message_class)
                pose = msg.pose if message_class is PoseStamped else msg.pose.pose
                p = pose.position
                if not np.all(np.isfinite([p.x, p.y])):
                    raise ValueError(f'{topic}: nonfinite position')
                rows[topic].append((timestamp, p.x, p.y, msg.header.frame_id))
    if origin is None:
        raise ValueError(f'No bag messages found in {bag}')
    series = {}
    for topic, values in rows.items():
        if not values:
            print(f'WARNING: {topic} has no recorded messages; its curve will be absent.')
            continue
        values.sort(key=lambda row: row[0])
        frames = {row[3] for row in values}
        if len(frames) != 1:
            raise ValueError(f'{topic}: frame changed during recording: {frames}')
        time = np.array([(row[0] - origin) / 1e9 for row in values])
        position = np.array([[row[1], row[2]] for row in values])
        keep = np.r_[True, np.diff(time) > 0]
        series[topic] = (time[keep], position[keep], next(iter(frames)))
        print(f'{topic}: {len(values)} poses, frame={next(iter(frames))}')
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, default=DEFAULT_BAG, help='ROS bag directory containing .db3 files')
    parser.add_argument('--output', type=Path, help='Output PNG path')
    parser.add_argument('--start', type=float, help='Comparison start in seconds from bag start')
    parser.add_argument('--end', type=float, help='Comparison end in seconds from bag start')
    parser.add_argument('--no-show', action='store_true', help='Save without opening a plot window')
    args = parser.parse_args()
    series = read_positions(args.bag)
    if not series:
        parser.error('No position topics found')
    start = max(t[0] for t, _, _ in series.values()) if args.start is None else args.start
    end = min(t[-1] for t, _, _ in series.values()) if args.end is None else args.end
    if end <= start or any(start < t[0] or end > t[-1] for t, _, _ in series.values()):
        parser.error('Comparison interval must lie within every available source recording')
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for topic, (time, position, frame) in series.items():
        label = TOPICS[topic][0]
        mask = (time >= start) & (time <= end)
        if mask.sum() < 2:
            parser.error(f'Too few {label} samples in comparison interval')
        initial = np.array([np.interp(start, time, position[:, j]) for j in range(2)])
        displacement = np.linalg.norm(position[mask] - initial, axis=1)
        axes[0].plot(time[mask], displacement, label=label)
        for j in range(2):
            axes[j + 1].plot(time[mask], position[mask, j], label=f'{label} ({frame})')
    axes[0].set_ylabel('XY displacement (m)')
    axes[0].set_title(f'Position comparison: {args.bag.parent.name}/{args.bag.name}')
    axes[1].set_ylabel('Recorded X (m)')
    axes[2].set_ylabel('Recorded Y (m)')
    axes[2].set_xlabel('Bag recording time (s)')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.3)
    notes = 'Raw X/Y use different frames; no frame transform or time-lag correction applied.'
    if '/odometry/filtered' not in series:
        notes += '\nEKF unavailable: no recorded /odometry/filtered poses.'
    fig.text(.5, .01, notes, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    output = args.output or REPO / 'figs' / args.bag.parent.name / args.bag.name / 'position_comparison.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f'Saved: {output.resolve()}')
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
