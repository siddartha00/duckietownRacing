#!/usr/bin/env python3
import torch
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np

class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.VISUALIZATION
        )
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'duckiebot12')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()

        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self.seg_pub = rospy.Publisher(
            f"/{self._vehicle_name}/segmentation/image/compressed",
            CompressedImage,
            queue_size=1
        )
        print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def callback(self, msg):
        # Convert ROS image to OpenCV
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # --- White mask ---
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

        # --- Yellow mask ---
        mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))

        # --- Road mask (dark areas) ---
        mask_road = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 120]))

        # --- Only bottom half for road ---
        h, w = mask_road.shape
        mask_road[:h//2, :] = 0  # zero out top half

        # --- Morphological operations to remove noise ---
        kernel = np.ones((5,5), np.uint8)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, kernel)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, kernel)

        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        # --- Combine masks into colored image ---
        mask_color = np.zeros_like(image)
        mask_color[mask_road > 0] = [0, 0, 255]       # red for road
        mask_color[mask_white > 0] = [255, 255, 255]  # white lanes
        mask_color[mask_yellow > 0] = [0, 255, 255]   # yellow lanes

        # Publish the mask
        seg_msg = self._bridge.cv2_to_compressed_imgmsg(mask_color)
        self.seg_pub.publish(seg_msg)

if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()
