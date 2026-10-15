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
- `depth_topic` (`sensor_msgs/Image`) - depth registered to RGB, `16UC1` in millimetres or `32FC1` in metres
- `camera_info_topic` (`sensor_msgs/CameraInfo`) — matching RGB camera intrinsics
- `odom_topic` (`nav_msgs/Odometry`) — robot pose, default `/odometry/local`

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
  odom_topic:=/odometry/local \
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

### Required RGB/depth alignment

The full weed bringup above explicitly enables software depth-to-colour
registration in the included Orbbec camera launch:

```text
depth_registration:=true
align_mode:=SW
align_target_stream:=COLOR
```

These settings are supplied automatically; operators do not need to add them to
the full weed bringup command. RGB bounding boxes must be paired with registered
depth. Resizing an unregistered depth image does not align the sensor viewpoints.

The detector/geotagger-only launch does not start or configure a camera. If the
camera is already running as part of another system, apply the same settings in
that camera launch and reuse its streams. Do not start a second camera driver.
For a standalone camera test:

```bash
ros2 launch orbbec_camera gemini_330_series.launch.py \
  camera_name:=camera \
  enable_color:=true \
  enable_depth:=true \
  depth_registration:=true \
  align_mode:=SW \
  align_target_stream:=COLOR
```

With the tested Orbbec driver, the registered output remains
`/camera/depth/image_raw`; `/camera/depth/image_unaligned` is not suitable for
the geotagger. Use `/camera/color/camera_info` for the matching RGB intrinsics.
After changing camera drivers or profiles, verify the output topics, colour
optical frame, image dimensions and calibration, then visually check alignment
at plant boundaries. Matching metadata alone is not an alignment-quality test.


## Live RViz map

With the robot stack and detector/geotagger running on the Jetson, run this in a
local desktop terminal (not the SSH terminal), after building and sourcing this
package on the desktop:

```bash
ros2 launch weed_geotagging weed_visualisation_launch.py
```

The desktop must use the same ROS domain and discovery configuration as the
Jetson. This launch starts only RViz and the odometry visualiser; it does not
start another camera, motor driver or YOLO instance.

The saved top-down view shows a metre grid, a rover box with a red heading arrow,
a cyan route, and confirmed plant markers labelled with the YOLO class and stable
track ID. Classes have deterministic colours; different classes can share a
colour, so the text label identifies the type. A model trained with only one
weed class cannot identify additional species. Enable the optional Camera display
for RGB, or change its topic to `/weed_detections/image` when annotated publishing
is enabled.

The rover shape is schematic. Set `rover_length_m`, `rover_width_m`, and
`rover_height_m` to adjust its size. The route samples positions every 0.03 m and
keeps at most 10000 poses by default; set `path_spacing_m` and `max_path_points`
to change this. Route history starts when the desktop visualiser starts and
resets on restart. Use `start_rviz:=false` to publish the rover displays without
opening a window.

Topics: `/rover_markers` (MarkerArray), `/rover_path` (Path), and `/weed_markers`
(MarkerArray). Plant snapshots clear obsolete markers and labels after track
merges, and are retained for RViz opened later. The display uses the odometry
header frame; the saved RViz fixed frame defaults to `odom`. Change Fixed Frame
and the view's Target Frame if using a differently named local EKF frame.

## Runtime dependencies

The ROS package declares the ROS-side dependencies. The YOLO node also needs
Ultralytics installed in the same Python environment used by ROS:

```bash
python3 -m pip install ultralytics
```
