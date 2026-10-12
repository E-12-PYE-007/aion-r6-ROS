# weed_geotagging

Projects weed detections from image space into the rover odometry/map frame.

## YOLO detection

`yolo_weed_detector` subscribes to the RGB camera topic, runs an Ultralytics YOLO
model such as `best.pt`, and publishes `std_msgs/String` JSON detections on
`/weed_detections_json`.

The model weights are passed as a launch parameter instead of being stored in
this repository:

```bash
ros2 launch weed_geotagging yolo_weed_detector_launch.py model_path:=/home/vla-cap/models/best.pt
```

Optional useful args:

```bash
confidence:=0.25 iou:=0.45 device:=0 target_classes:=weed publish_annotated:=true
```

The detector publishes annotated images on `/weed_detections/image` when
`publish_annotated` is true.

## Detection adapter

Accepted JSON examples:

```json
{"detections": [{"bbox": [300, 220, 340, 260], "class": "weed", "score": 0.95}]}
```

```json
[{"xmin": 300, "ymin": 220, "xmax": 340, "ymax": 260, "label": "weed", "confidence": 0.95}]
```

## Inputs

- `image_topic` (`sensor_msgs/Image`) - RGB image for YOLO
- `detections_topic` (`std_msgs/String`) - JSON detections
- `depth_topic` (`sensor_msgs/Image`) - `16UC1` in millimetres or `32FC1` in metres
- `camera_info_topic` (`sensor_msgs/CameraInfo`) — camera intrinsics
- `odom_topic` (`nav_msgs/Odometry`) — robot pose, default `/odometry/filtered`

## Outputs

- `weed_markers` (`visualization_msgs/MarkerArray`) — persistent weed tracks in `odom`
- Optional CSV via `output_csv`

The geotagger now keeps a small in-memory track list so the same plant is not
recorded as a new weed every frame. A new measurement is matched to an existing
track when it is within `merge_radius_m`; otherwise a new `track_id` is created.
Existing tracks are updated with a weighted moving average based on YOLO score,
depth quality, distance, and current angular speed.

The CSV includes both raw measurement columns and smoothed track columns. Useful
accuracy/debug columns include `track_id`, `event`, `observations`,
`raw_weed_x/y/z`, `track_weed_x/y/z`, `depth_std_m`, `depth_inlier_samples`,
`depth_valid_ratio`, `match_distance_m`, `linear_x`, and `angular_z`.

## Run

Run detection and geotagging together:

```bash
ros2 launch weed_geotagging weed_detection_geotagging_launch.py \
  model_path:=/home/vla-cap/models/best.pt \
  output_csv:=/home/vla-cap/weeds.csv \
  odom_topic:=/odometry/filtered \
  publish_annotated:=false
```

Useful tracking/depth tuning args:

```bash
merge_radius_m:=0.35 bbox_shrink_ratio:=0.2 max_depth_samples:=800 min_depth_samples:=20
```

`target_label` defaults to empty, which accepts every YOLO class. Set it if you
only want one class, for example `target_label:=plant2`.

Or run only the geotagger if another node is already publishing detections:

```bash
ros2 launch weed_geotagging weed_geotagging_launch.py output_csv:=/tmp/weeds.csv
```

For a local smoke test with fake data:

```bash
ros2 run weed_geotagging fake_weed_inputs
ros2 launch weed_geotagging weed_geotagging_launch.py output_csv:=/tmp/weeds.csv
```

## Runtime dependencies

The ROS package declares the ROS-side dependencies. The YOLO node also needs
Ultralytics installed in the same Python environment used by ROS:

```bash
python3 -m pip install ultralytics
```
