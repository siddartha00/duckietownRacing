#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped

class UVAlignControllerNode(DTROS):
    def __init__(self, node_name):
        super(UVAlignControllerNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

     
        self.bridge = CvBridge()
        self.vehicle_name = os.environ['VEHICLE_NAME']
        self.image_topic = f"/{self.vehicle_name}/camera_node/image/compressed"
        self.cmd_topic = f"/{self.vehicle_name}/car_cmd_switch_node/cmd"

      
        self.u_desired = 330 
        self.v_desired = 240  

        self.center_threshold = 5     
        self.angular_gain = 0.005      

        self.publisher = rospy.Publisher(self.cmd_topic, Twist2DStamped, queue_size=1)
        rospy.Subscriber(self.image_topic, CompressedImage, self.camera_callback)

    def camera_callback(self, msg):
       
        image = self.bridge.compressed_imgmsg_to_cv2(msg)
        height, width, _ = image.shape

      
        u = width // 2
        v = height // 2

        error_u = u - self.u_desired

        if abs(error_u) > self.center_threshold:
            omega = -self.angular_gain * error_u  
        else:
            omega = 0.0

        move = Twist2DStamped(v=0.0, omega=omega)
        self.publisher.publish(move)

        rospy.loginfo(f"[u={u}] Target={self.u_desired}, Error={error_u}, Omega={omega:.3f}")

    def on_shutdown(self):
        print("stopping")
        stop = Twist2DStamped(v=0.0, omega=0.0)
        self.publisher.publish(stop)

if __name__ == '__main__':
    node = UVAlignControllerNode(node_name='uv_align_controller_node')
    rospy.spin()
