"""Regression tests for map snapshots, track merges, and rover routes."""

from copy import deepcopy
import unittest
from unittest.mock import Mock

import rclpy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

from weed_geotagging.rover_visualiser import RoverVisualiser
from weed_geotagging.weed_geotagger import WeedGeotagger, WeedTrack


def odometry(x=0.0, y=0.0, frame='odom'):
    msg = Odometry()
    msg.header.frame_id = frame
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.w = 1.0
    return msg


class VisualisationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init(args=[])

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_merged_track_and_label_are_removed_from_snapshot(self):
        node = WeedGeotagger()
        self.addCleanup(node.destroy_node)
        node.marker_pub = Mock()
        node.latest_odom = odometry(frame='local_odom')
        node.tracks = [
            WeedTrack(1, 'broadleaf', 1.0, 0.0, 0.0, 0.9, 3, 0.0, 1.0, 0.2, 0.0, 3),
            WeedTrack(2, 'broadleaf', 1.1, 0.0, 0.0, 0.9, 3, 0.0, 1.0, 0.2, 0.0, 3),
        ]
        node.publish_track_markers()
        self.assertEqual(len(node.marker_pub.publish.call_args.args[0].markers), 5)
        node.merge_nearby_tracks()
        node.publish_track_markers()
        snapshot = node.marker_pub.publish.call_args.args[0].markers
        self.assertEqual(snapshot[0].action, Marker.DELETEALL)
        self.assertEqual(len(snapshot), 3)
        self.assertEqual({marker.id for marker in snapshot[1:]}, {1})
        self.assertEqual(snapshot[2].text, 'broadleaf | ID 1')
        self.assertEqual(snapshot[2].type, Marker.TEXT_VIEW_FACING)
        self.assertEqual(snapshot[1].header.frame_id, 'local_odom')
        for channel in ('r', 'g', 'b'):
            self.assertEqual(getattr(snapshot[1].color, channel), getattr(snapshot[2].color, channel))

    def test_empty_snapshot_clears_previous_map_and_candidates_are_hidden(self):
        node = WeedGeotagger()
        self.addCleanup(node.destroy_node)
        node.marker_pub = Mock()
        node.tracks = [WeedTrack(1, 'weed', 1.0, 0.0, 0.0, 0.9, 1, 0.0, 1.0, 1.0, 0.0, 3)]
        node.publish_track_markers()
        snapshot = node.marker_pub.publish.call_args.args[0].markers
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(snapshot[0].action, Marker.DELETEALL)

    def test_class_colours_are_stable(self):
        self.assertEqual(WeedGeotagger.class_colour('grass'), WeedGeotagger.class_colour('grass'))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in WeedGeotagger.class_colour('grass')))

    def test_turn_updates_rover_pose_without_duplicate_route_points(self):
        node = RoverVisualiser()
        self.addCleanup(node.destroy_node)
        node.rover_pub = Mock()
        node.path_pub = Mock()
        msg = odometry(1.0, 2.0)
        node.odom_callback(msg)
        turn = deepcopy(msg)
        turn.pose.pose.orientation.z = 1.0
        turn.pose.pose.orientation.w = 0.0
        node.odom_callback(turn)
        node.publish_visualisation()
        self.assertEqual(len(node.path_poses), 1)
        body, heading = node.rover_pub.publish.call_args.args[0].markers
        self.assertEqual(body.type, Marker.CUBE)
        self.assertEqual(heading.type, Marker.ARROW)
        self.assertEqual(heading.pose.orientation, turn.pose.pose.orientation)
        self.assertEqual(msg.pose.pose.position.z, 0.0)

    def test_route_is_bounded_and_resets_when_frame_changes(self):
        from collections import deque
        node = RoverVisualiser()
        self.addCleanup(node.destroy_node)
        node.path_poses = deque(maxlen=3)
        for x in range(5):
            node.odom_callback(odometry(float(x)))
        self.assertEqual([pose.pose.position.x for pose in node.path_poses], [2.0, 3.0, 4.0])
        node.odom_callback(odometry(8.0, frame='other_odom'))
        self.assertEqual(len(node.path_poses), 1)
        self.assertEqual(node.path_poses[0].header.frame_id, 'other_odom')


if __name__ == '__main__':
    unittest.main()
