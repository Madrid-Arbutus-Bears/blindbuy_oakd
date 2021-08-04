#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Point, PointStamped

import os

from blindbuy_interfaces.srv import ProductDistance


import time

from blindbuy_interfaces.action import LocalNavigation

from blindbuy_interfaces.srv import ProductDistance

# import PyOpenAL (will require an OpenAL shared library)
from openal import *

from ament_index_python.packages import get_package_share_directory

import numpy as np

class LocalNavigation(Node):

    def __init__(self):
        super().__init__('local_navigation_server')
        self.product_distance_client = self.create_client(ProductDistance, 'product_distance')
        self.product_position_sub = self.create_subscription(PointStamped,'product_position',self.product_position_callback,1)
        #Obtain path for Sound file
        root_path=os.path.join(get_package_share_directory('blindbuy_oakd'), 'data/audio')
        # Set sound properties
        self.source = oalOpen(os.path.join(root_path,'beep.wav'))
        self.source.set_max_distance(5.0)
        self.source.set_position((0,0,0))
        #Declare openal listener
        self.listener=Listener()
        #Product position
        self.product_x=0.0
        self.product_y=0.0
        self.product_z=0.0

        while rclpy.ok():
            product_position_msg=Point()
            product_position_msg.x=self.product_x
            product_position_msg.y=self.product_y
            product_position_msg.z=self.product_z

            req = ProductDistance.Request()
            req.product_position=product_position_msg
            req.source_frame='head'

            while not self.product_distance_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Product Distance Server not available, waiting again...')

            future = self.product_distance_client.call_async(req)

            rclpy.spin_until_future_complete(self, future)

            try:
                result = future.result()
                distance=result.distance
                self.listener.set_position((result.frame_position.x,result.frame_position.y,result.frame_position.z))
                orientation=result.transform.transform.rotation
                openal_orientation = self.quaternion_to_openal(orientation)
                self.listener.set_orientation(openal_orientation) #https://stackoverflow.com/questions/7861306/clarification-on-openal-listener-orientation
                self.source.set_position((product_position_msg.x, product_position_msg.y, product_position_msg.z))
                time.sleep(distance/4)
                self.source.play()
                
            except Exception as e:
                self.get_logger().warning('Service call failed %r' % (e,))

    def quaternion_to_openal(self, orientation):
        #https://math.stackexchange.com/questions/2618527/converting-from-yaw-pitch-roll-to-vector
        #https://math.stackexchange.com/questions/2253071/convert-quaternion-to-vector/2253214

        #Calculate 'up vector' from 'at vector'. Perpendicular vector across y axis: https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis
        at_vector = np.array([orientation.x,  orientation.y,  orientation.z])

        # Compute the rotation matrix
        R = self.get_rotation_matrix(at_vector, [0.0, 1.0, 0.0])

        # Apply the rotation matrix to the vector
        up_vector = np.dot(at_vector.T, R.T)     

        openal_orientation=(at_vector[0],at_vector[1],at_vector[2],up_vector[0],up_vector[1],up_vector[2])

        return openal_orientation
    
    def get_rotation_matrix(self, i_v, unit=None):
        # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
        if unit is None:
            unit = [1.0, 0.0, 0.0]
        # Normalize vector length
        i_v /= np.linalg.norm(i_v)

        # Get axis
        uvw = np.cross(i_v, unit)

        # Compute trig values - no need to go through arccos and back
        rcos = np.dot(i_v, unit)
        rsin = np.linalg.norm(uvw)

        #Normalize and unpack axis
        if not np.isclose(rsin, 0):
            uvw /= rsin
        u, v, w = uvw

        # Compute rotation matrix - re-expressed to show structure
        return (
            rcos * np.eye(3) +
            rsin * np.array([
                [ 0, -w,  v],
                [ w,  0, -u],
                [-v,  u,  0]
            ]) +
            (1.0 - rcos) * uvw[:,None] * uvw[None,:]
        )
    
    def product_position_callback(self, msg):
        self.product_x=msg.point.x
        self.product_y=msg.point.y
        self.product_z=msg.point.z

def main(args=None):
    rclpy.init(args=args)

    executor = MultiThreadedExecutor(num_threads=8)

    local_navigation = LocalNavigation()

    executor.add_node(local_navigation)

    executor.spin()

    executor.shutdown()

    # release resources (don't forget this)
    oalQuit()

if __name__ == '__main__':
    main()