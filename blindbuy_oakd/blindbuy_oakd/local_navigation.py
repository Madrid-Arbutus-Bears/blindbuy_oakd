#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType

from geometry_msgs.msg import Point, PointStamped

import os

from blindbuy_interfaces.srv import ProductDistance

from rclpy.action import ActionServer, ActionClient

import time

from blindbuy_interfaces.action import LocalNavigation, Sound

from blindbuy_interfaces.srv import ProductDistance

from tf2_ros import TransformBroadcaster, TransformListener, TransformStamped, Buffer

# import PyOpenAL (will require an OpenAL shared library)
from openal import *

from ament_index_python.packages import get_package_share_directory



class LocalNavigation(Node):

    def __init__(self):
        super().__init__('local_navigation_server')
        self.product_distance_client = self.create_client(ProductDistance, 'product_distance')
        self.product_position_sub = self.create_subscription(PointStamped,'product_position',self.product_position_callback,1)
        #Timer
        # timer_period = 0.033  # 30Hz
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        #Obtain path for Sound file
        root_path=os.path.join(get_package_share_directory('blindbuy'), 'data/audio')
        # Set sound properties
        self.source = oalOpen(os.path.join(root_path,'bounce.wav'))
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

            print(product_position_msg)

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
                #self.listener.set_orientation((0,0,-1,0,1,0)) #https://stackoverflow.com/questions/7861306/clarification-on-openal-listener-orientation
                self.source.set_position((product_position_msg.x, product_position_msg.y, product_position_msg.z))
                time.sleep(distance/4)
                self.source.play()
                
            except Exception as e:
                self.get_logger().warning('Service call failed %r' % (e,))

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


