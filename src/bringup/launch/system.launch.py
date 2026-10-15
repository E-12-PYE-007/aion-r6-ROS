"""Full rover system.

  rover base (mavros, motor control, local localisation)
  + camera + ag_vla (cam_seq_bridge, sys1_hw)
  + controller (CBF safety layer or pure pursuit)
  + global localisation (optional) + weed detection and geotagging (optional)

nvblox runs separately in the Isaac ROS docker, and the VLA's sys2 on the workstation
(matching ROS_DOMAIN_ID). Weed visualisation is a workstation launch:
weed_geotagging weed_visualisation.launch.py.

Arguments: controller (cbf | pure_pursuit), localisation (local | global), weed,
model_path, output_csv, use_sim_time.
"""
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression

CHOICES = {
    'controller': ('cbf', 'pure_pursuit'),
    'localisation': ('local', 'global'),
}


def include(package, launch_file, condition=None, **launch_arguments):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            get_package_share_directory(package) + '/launch/' + launch_file),
        launch_arguments=launch_arguments.items(),
        condition=condition,
    )


def when(argument, value):
    return IfCondition(PythonExpression(["'", LaunchConfiguration(argument), f"' == '{value}'"]))


def validate(context):
    for name, allowed in CHOICES.items():
        value = LaunchConfiguration(name).perform(context)
        if value not in allowed:
            raise RuntimeError(f"{name}:={value} is not valid, use one of {allowed}")
    return []


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='cbf',
                              description='cbf (safety layer) or pure_pursuit'),
        DeclareLaunchArgument('localisation', default_value='local',
                              description='local, or global (adds navsat and the global EKF)'),
        DeclareLaunchArgument('weed', default_value='true',
                              description='Run weed detection and geotagging'),
        DeclareLaunchArgument('model_path', default_value='/home/vla-cap/YOLO_release/best.pt',
                              description='YOLO weights for weed detection'),
        DeclareLaunchArgument('output_csv', default_value='',
                              description='Weed geotag CSV path (empty: no CSV)'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation clock'),
        OpaqueFunction(function=validate),

        include('bringup', 'rover_base.launch.py', use_sim_time=use_sim_time),
        include('bringup', 'camera.launch.py'),
        include('ag_vla', 'ag_vla.launch.py'),

        include('control', 'safety_layer.launch.py', condition=when('controller', 'cbf')),
        include('control', 'pure_pursuit.launch.py', condition=when('controller', 'pure_pursuit')),

        include('localisation', 'global_localisation.launch.py',
                condition=when('localisation', 'global'), use_sim_time=use_sim_time),

        include('weed_geotagging', 'weed_detection_geotagging.launch.py',
                condition=IfCondition(LaunchConfiguration('weed')),
                model_path=LaunchConfiguration('model_path'),
                output_csv=LaunchConfiguration('output_csv'),
                publish_annotated='false',
                max_depth_m='3.0',
                foreground_depth_percentile='40.0',
                foreground_depth_tolerance_m='0.4',
                turning_merge_radius_m='0.9',
                track_merge_radius_m='0.5'),
    ])
