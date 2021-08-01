#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from blindbuy_interfaces.srv import ProductDistance

from tf2_ros import TransformListener, Buffer

from geometry_msgs.msg import Point

import math

from rclpy.time import Time

import numpy as np
import rclpy
from rclpy.node import Node

#ros2 service call /product_distance blindbuy_interfaces/srv/ProductDistance "source_frame: 'head'"
#ros2 service call /product_distance blindbuy_interfaces/srv/ProductDistance "{source_frame: 'head', product_position: {x: 1.0, y: 0.0, z: 0.0}}"

class ProductDistanceServer(Node):

    def __init__(self):
        super().__init__('product_distance_server')
        self.srv = self.create_service(ProductDistance, 'product_distance', self.product_distance_callback)
        self.tfBuffer = Buffer()
        self.listener = TransformListener(self.tfBuffer, self)
        self.get_logger().info("Product Distance Server is ready")
    
    def distance(self, P1, P2):     
        dist = math.sqrt(math.pow(P2.x - P1.x, 2) +
                    math.pow(P2.y - P1.y, 2) +
                    math.pow(P2.z - P1.z, 2)* 1.0) 
        return dist

    def product_distance_callback(self, request, response):
        try:
            trans = self.tfBuffer.lookup_transform('base_link', request.source_frame, Time())

            #Doing this because TransformStamped return Vector3 instead of Point
            frame_position=Point()
            frame_position.x=trans.transform.translation.x
            frame_position.y=trans.transform.translation.y
            frame_position.z=trans.transform.translation.z

            distance=self.distance(request.product_position, frame_position)

            print(distance)
            print(request.product_position)
            print(frame_position)

            #Find 3D distance -> https://stackoverflow.com/questions/20184992/finding-3d-distances-using-an-inbuilt-function-in-python
            # p0 = np.array([request.product_position.x,request.product_position.y,0.0])
            # p1 = np.array([trans.transform.translation.x,trans.transform.translation.y,0.0])
            # distance = np.linalg.norm(p0 - p1)



            # distance=math.dist([request.product_position.x,request.product_position.y],[trans.transform.translation.x,trans.transform.translation])
            #distance=math.dist([1,2],[2,3])

            #self.get_logger().info("Distance: %s -> product=(%s,%s,%s) target=(%s,%s,%s)"%(distance,p0[0],p0[1],p0[2],p1[0],p1[1],p1[2]))
            
            response=ProductDistance.Response()
            response.transform=trans
            response.distance=distance
            response.frame_position=frame_position
            
            return response
            
        except Exception as ex:
            template = "Error while computing transform. An exception of type {0} occurred: {1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.get_logger().error(message)
            
            response=ProductDistance.Response()

            return response
        


def main(args=None):
    rclpy.init(args=args)

    product_distance_server = ProductDistanceServer()

    rclpy.spin(product_distance_server)

    rclpy.shutdown()


if __name__ == '__main__':
    main()