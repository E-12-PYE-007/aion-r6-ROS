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

- `weed_markers` (`visualization_msgs/MarkerArray`) — weed points in `odom`
- Optional CSV via `output_csv`

## Run

Run detection and geotagging together:

```bash
ros2 launch weed_geotagging weed_detection_geotagging_launch.py \
  model_path:=/home/vla-cap/models/best.pt \
  output_csv:=/home/vla-cap/weeds.csv \
  odom_topic:=/odometry/filtered
```

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
