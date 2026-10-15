import cv2
import numpy as np
from std_msgs.msg import Header

from weed_geotagging.yolo_weed_detector import to_compressed_image


def test_to_compressed_image_is_a_decodable_jpeg_with_the_source_header():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[10:30, 20:50] = (0, 200, 0)
    header = Header(frame_id='camera_color_optical_frame')
    header.stamp.sec = 12
    header.stamp.nanosec = 345

    msg = to_compressed_image(image, header)

    assert msg.format == 'jpeg'
    assert msg.header.frame_id == 'camera_color_optical_frame'
    assert (msg.header.stamp.sec, msg.header.stamp.nanosec) == (12, 345)
    decoded = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == image.shape
