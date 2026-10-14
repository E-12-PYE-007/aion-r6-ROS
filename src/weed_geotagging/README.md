# weed_geotagging

Projects weed detections from image space into the rover odometry/map frame.

## YOLO detection

`yolo_weed_detector` subscribes to the RGB camera topic, runs an Ultralytics YOLO
model such as `best.pt`, and publishes `std_msgs/String` JSON detections on
`/weed_detections_json`.

The model weights are passed as a launch parameter instead of being stored in
this repository. The main launch starts both YOLO and geotagging:

```bash
ros2 launch weed_geotagging weed_detection_geotagging_launch.py \
  model_path:=/home/vla-cap/models/best.pt
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
- Optional detection CSV via `output_csv`
- Optional continuous odometry CSV via `output_odom_csv`. If `output_csv` is set
  and `output_odom_csv` is left empty, the geotagger automatically writes a
  sibling file named `<output_csv_stem>_odom.csv`.

The geotagger now keeps a small in-memory track list so the same plant is not
recorded as a new weed every frame. A new measurement is matched to an existing
track when it is within `merge_radius_m`; otherwise a new `track_id` is created.
Existing tracks are updated with a weighted moving average based on YOLO score,
depth quality, distance, and current angular speed.
The geotagger buffers recent depth frames and odometry, then projects each
detection using the depth and odom messages closest to the original image
timestamp. If either one is farther than `max_sync_age_sec`, that detection is
skipped instead of being geotagged with stale data. This is important because
YOLO inference can publish detections after the rover has already moved or
turned. During turns it uses a larger `turning_merge_radius_m` for matching
existing tracks and requires `turning_min_confirmed_observations` before
confirming turn-born candidates. Nearby duplicate tracks are also merged with
`track_merge_radius_m`.

The CSV includes both raw measurement columns and smoothed track columns. Useful
accuracy/debug columns include `track_id`, `event`, `observations`,
`raw_weed_x/y/z`, `track_weed_x/y/z`, `depth_std_m`, `depth_inlier_samples`,
`depth_valid_ratio`, `depth_stamp_sec`, `odom_stamp_sec`, `odom_age_sec`,
`match_distance_m`, `linear_x`, and `angular_z`.

For tape-measured tests from the rover body, compare against
`robot_forward_m` and `robot_left_m` (`x_robot` and `y_robot`). For local EKF
map consistency while driving, use `local_ekf_x_m` and `local_ekf_y_m`
(`track_weed_x` and `track_weed_y`).

## Run

Run detection and geotagging together:

```bash
ros2 launch weed_geotagging weed_detection_geotagging_launch.py \
  model_path:=/home/vla-cap/models/best.pt \
  output_csv:=/home/vla-cap/weeds.csv \
  output_odom_csv:=/home/vla-cap/weeds_odom.csv \
  odom_topic:=/odometry/filtered \
  publish_annotated:=false
```

Useful tracking/depth tuning args:

```bash
merge_radius_m:=0.35 bbox_shrink_ratio:=0.2 max_depth_m:=3.0
foreground_depth_percentile:=20.0 foreground_depth_tolerance_m:=0.6
turn_tracking_angular_z:=0.15 turning_merge_radius_m:=0.75
min_confirmed_observations:=3 turning_min_confirmed_observations:=6
track_merge_radius_m:=0.25 odom_buffer_sec:=10.0
depth_buffer_sec:=10.0 max_sync_age_sec:=0.1
max_depth_samples:=800 min_depth_samples:=20
```

Depth sampling is foreground-biased: it samples many pixels inside the shrunken
bbox, rejects invalid/far depth, estimates the weed distance from a near-depth
percentile, then keeps pixels close to that foreground depth. This helps avoid
using background depth behind thin plants.

`target_label` defaults to empty, which accepts every YOLO class. Set it if you
only want one class, for example `target_label:=plant2`.

For a full Jetson-side rover launch, including depth camera, local EKF,
`roboclaw_for_motors`, `cmd_vel_to_roboclaw`, YOLO, and geotagging:

```bash
ros2 launch bringup weed_geotagging_bringup_launch.py \
  model_path:=/home/vla-cap/models/best.pt \
  output_csv:=/home/vla-cap/weeds.csv
```


## Runtime dependencies

The ROS package declares the ROS-side dependencies. The YOLO node also needs
Ultralytics installed in the same Python environment used by ROS:

```bash
python3 -m pip install ultralytics
```
