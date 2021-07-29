#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType

from geometry_msgs.msg import Point

import os

from blindbuy_interfaces.srv import ProductDistance

from rclpy.action import ActionServer, ActionClient

import time

from blindbuy_interfaces.action import LocalNavigation, Sound

from blindbuy_interfaces.srv import ProductDistance

from tf2_ros import TransformBroadcaster, TransformListener, TransformStamped, Buffer

import pymongo

# import PyOpenAL (will require an OpenAL shared library)
from openal import * 

from ament_index_python.packages import get_package_share_directory

#ros2 action send_goal local_navigation blindbuy_interfaces/action/LocalNavigation "{barcode: 8480000152039}"

class LocalNavigationServer(Node):

    def __init__(self):
        super().__init__('local_navigation__server')
        self._action_server = ActionServer(self,LocalNavigation,'local_navigation',self.execute_callback)  
        client = pymongo.MongoClient("mongodb+srv://manager:blindbuy97@cluster0.oiz2e.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
        self.product_distance_client = self.create_client(ProductDistance, 'product_distance')
        self.sound_client = ActionClient(self, Sound, 'sound')
        self.db = client.demoshop
         #Obtain path for Sound file
        root_path=os.path.join(get_package_share_directory('blindbuy'), 'data/audio')
        # Set sound properties
        self.source = oalOpen(os.path.join(root_path,'bounce.wav'))
        self.source.set_max_distance(5.0)
        self.source.set_position((0,0,0))
        #Declare listener
        self.listener=Listener()


    def execute_callback(self, goal_handle):
        barcode=goal_handle.request.barcode
        product_db=self.db.products.find_one({"_id" : barcode})
        if not product_db:
            self.get_logger().warning("No product found in the database with barcode '%s'"%(barcode))
            goal_handle.abort()
        product_position=product_db.get('product_position')

        product_position_msg=Point()
        product_position_msg.x=product_position.get('x')
        product_position_msg.y=product_position.get('y')
        product_position_msg.z=product_position.get('z')

        print(product_position_msg)

        req = ProductDistance.Request()
        req.product_position=product_position_msg
        req.source_frame='head'

        distance=100 #Initialization to a high number

        while True:
            while not self.product_distance_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Product Distance Server not available, waiting again...')

            future = self.product_distance_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            try:
                result = future.result()
            except Exception as e:
                self.get_logger().warning('Service call failed %r' % (e,))
            
            distance=result.distance
            self.listener.set_position((result.frame_position.x,result.frame_position.y,result.frame_position.z))
            #self.listener.set_orientation((0,0,-1,0,1,0)) #https://stackoverflow.com/questions/7861306/clarification-on-openal-listener-orientation
            self.source.set_position((product_position_msg.x, product_position_msg.y, product_position_msg.z))
            self.source.play()
            time.sleep(distance/4)
        
 
            
def main(args=None):
    rclpy.init(args=args)

    local_navigation_server = LocalNavigationServer()

    rclpy.spin(local_navigation_server)

    # release resources (don't forget this)
    oalQuit()


if __name__ == '__main__':
    main()