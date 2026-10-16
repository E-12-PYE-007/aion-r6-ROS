#!/usr/bin/env python3
"""Render a system rosbag into a two-panel MP4: the odom-frame map (robot path +
deduplicated weed tracks) on the left, the annotated camera feed on the right.

Weed positions come straight from /weed_markers, which is the weed_geotagger's
already-deduplicated, confirmed-track output -- this tool does no merging of its own.
The right-hand video is the raw per-frame YOLO overlay, so the same weed is boxed in
many frames there while the map shows each confirmed track once.

Run with ROS 2 sourced (needs rosbag2_py and the standard message types):

    source /opt/ros/humble/setup.bash
    python3 tools/weed_bag_video.py /home/vla-cap/final_rosbags/integ_test1

Only reads nav_msgs/Odometry, visualization_msgs/MarkerArray and
sensor_msgs/CompressedImage, so the custom_msgs package is not required.
"""
import argparse
import math
import os

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DELETEALL = 3  # visualization_msgs/Marker.DELETEALL


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def parse_weeds(marker_array):
    """One /weed_markers snapshot -> [(track_id, label, x, y)], ordered by id.

    'weeds' markers carry the position, 'weed_labels' the text; they share the track id.
    A snapshot is complete, so whatever is not in it is treated as gone.
    """
    positions, labels = {}, {}
    for m in marker_array.markers:
        if m.action == DELETEALL:
            continue
        if m.ns == "weeds":
            positions[m.id] = (m.pose.position.x, m.pose.position.y)
        elif m.ns == "weed_labels":
            labels[m.id] = m.text
    weeds = []
    for tid in sorted(positions):
        x, y = positions[tid]
        weeds.append((tid, labels.get(tid, f"ID {tid}"), x, y))
    return weeds


def read_bag(bag, odom_topic, markers_topic, image_topic):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for name in (odom_topic, markers_topic, image_topic):
        if name not in types:
            raise SystemExit(f"topic {name!r} not in bag; present: {sorted(types)}")

    odom, weeds, images = [], [], []
    odom_cls = get_message(types[odom_topic])
    markers_cls = get_message(types[markers_topic])
    image_cls = get_message(types[image_topic])
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == odom_topic:
            m = deserialize_message(data, odom_cls)
            p = m.pose.pose
            odom.append((t, p.position.x, p.position.y, yaw_from_quat(p.orientation)))
        elif topic == markers_topic:
            m = deserialize_message(data, markers_cls)
            weeds.append((t, parse_weeds(m)))
        elif topic == image_topic:
            m = deserialize_message(data, image_cls)
            frame = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                images.append((t, frame))
    if not odom:
        raise SystemExit("no odometry in bag")
    return odom, weeds, images


def latest_at(series, t):
    """Last (t_i, value) with t_i <= t, else None. series is time-ordered."""
    found = None
    for item in series:
        if item[0] <= t:
            found = item
        else:
            break
    return found


def render_map(ax, odom, weeds, t, limits):
    ax.cla()
    xs = [o[1] for o in odom if o[0] <= t]
    ys = [o[2] for o in odom if o[0] <= t]
    allx = [o[1] for o in odom]
    ally = [o[2] for o in odom]
    ax.plot(allx, ally, color="0.82", lw=1.0, zorder=1)          # full path, faint
    if xs:
        ax.plot(xs, ys, color="#1f77b4", lw=2.0, zorder=2)        # travelled so far
        cur = latest_at(odom, t)
        _, cx, cy, cyaw = cur
        ax.plot(cx, cy, "o", color="#1f77b4", ms=9, zorder=4)
        ax.annotate("", xy=(cx + 0.18 * math.cos(cyaw), cy + 0.18 * math.sin(cyaw)),
                    xytext=(cx, cy), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2))

    snap = latest_at(weeds, t)
    n = 0
    if snap:
        n = len(snap[1])
        for tid, label, wx, wy in snap[1]:
            ax.plot(wx, wy, "*", color="#2ca02c", ms=15, zorder=6)
            ax.annotate(label, (wx, wy), textcoords="offset points", xytext=(7, 5),
                        fontsize=8, color="#145a14", zorder=7)

    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, odom)")
    ax.set_ylabel("y (m, odom)")
    ax.set_title(f"Rover path + weeds    t={ (t - odom[0][0]) / 1e9:5.1f}s    weeds={n}")
    ax.grid(True, color="0.9")


def fig_to_bgr(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", default="/home/vla-cap/final_rosbags/integ_test1",
                    help="rosbag2 directory")
    ap.add_argument("--out", default=None, help="output mp4 (default: <bag>_viz.mp4)")
    ap.add_argument("--fps", type=int, default=15, help="output frame rate")
    ap.add_argument("--odom-topic", default="/odometry/local")
    ap.add_argument("--markers-topic", default="/weed_markers")
    ap.add_argument("--image-topic", default="/weed_detections/image/compressed")
    ap.add_argument("--panel-height", type=int, default=720, help="output panel height px")
    args = ap.parse_args()

    bag = args.bag.rstrip("/")
    out = args.out or (bag + "_viz.mp4")

    odom, weeds, images = read_bag(bag, args.odom_topic, args.markers_topic, args.image_topic)
    print(f"read {len(odom)} odom, {len(weeds)} weed snapshots, {len(images)} images")

    t0 = odom[0][0]
    t_end = max(odom[-1][0],
                weeds[-1][0] if weeds else t0,
                images[-1][0] if images else t0)
    duration_s = (t_end - t0) / 1e9

    # Map extent from the whole run (path + every weed ever seen), with a margin.
    px = [o[1] for o in odom] + [w[2] for _, snap in weeds for w in snap]
    py = [o[2] for o in odom] + [w[3] for _, snap in weeds for w in snap]
    margin = 0.4
    limits = (min(px) - margin, max(px) + margin, min(py) - margin, max(py) + margin)

    H = args.panel_height
    dpi = 100
    fig = plt.figure(figsize=(H / dpi, H / dpi), dpi=dpi)  # square map panel
    ax = fig.add_axes([0.13, 0.10, 0.84, 0.82])

    # Video panel width from the first frame's aspect, scaled to panel height H.
    sample = images[0][1] if images else np.zeros((H, H, 3), np.uint8)
    vid_w = int(round(sample.shape[1] * H / sample.shape[0]))
    vid_w += vid_w % 2  # keep even for the codec
    map_w = H
    map_w += map_w % 2
    out_w, out_h = map_w + vid_w, H

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"could not open VideoWriter for {out}")

    n_frames = int(duration_s * args.fps) + 1
    black = np.zeros((H, vid_w, 3), np.uint8)
    for i in range(n_frames):
        t = t0 + int(i / args.fps * 1e9)

        render_map(ax, odom, weeds, t, limits)
        map_bgr = fig_to_bgr(fig)
        if map_bgr.shape[:2] != (H, map_w):
            map_bgr = cv2.resize(map_bgr, (map_w, H))

        img = latest_at(images, t)
        vid = cv2.resize(img[1], (vid_w, H)) if img else black
        cv2.putText(vid, "annotated detections (raw per-frame)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(np.hstack([map_bgr, vid]))
        if i % 50 == 0:
            print(f"  frame {i}/{n_frames}")

    writer.release()
    plt.close(fig)
    print(f"wrote {out}  ({out_w}x{out_h}, {args.fps} fps, {duration_s:.1f}s)")


if __name__ == "__main__":
    main()
