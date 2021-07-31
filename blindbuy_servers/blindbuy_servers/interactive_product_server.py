#!/usr/bin/env python3
import sys

from interactive_markers import InteractiveMarkerServer
import rclpy
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker


def processFeedback(feedback):
    p = feedback.pose.position
    print(f'{feedback.marker_name} is now at {p.x}, {p.y}, {p.z}')

def normalizeQuaternion(quaternion_msg):
    norm = quaternion_msg.x**2 + quaternion_msg.y**2 + quaternion_msg.z**2 + quaternion_msg.w**2
    s = norm**(-0.5)
    quaternion_msg.x *= s
    quaternion_msg.y *= s
    quaternion_msg.z *= s
    quaternion_msg.w *= s

if __name__ == '__main__':
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('interactive_product_server')

    # create an interactive marker server on the namespace simple_marker
    server = InteractiveMarkerServer(node, 'product_marker')

    # create an interactive marker for our server
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = 'base_link'
    int_marker.name = 'product'

    # create a grey box marker
    box_marker = Marker()
    box_marker.type = Marker.MESH_RESOURCE
    box_marker.mesh_resource = "package://blindbuy_oakd/meshes/bottle.dae"
    box_marker.scale.x = 0.1
    box_marker.scale.y = 0.1
    box_marker.scale.z = 0.1
    box_marker.color.r = 0.0
    box_marker.color.g = 0.5
    box_marker.color.b = 0.7
    box_marker.color.a = 1.0

    # # create a non-interactive control which contains the box
    box_control = InteractiveMarkerControl()
    box_control.always_visible = True
    box_control.markers.append(box_marker)

    # # add the control to the interactive marker
    int_marker.controls.append(box_control)

    # # create a control which will move the box
    # # this control does not contain any markers,
    # # which will cause RViz to insert two arrows
    control = InteractiveMarkerControl()
    control.orientation.w = 1.0
    control.orientation.x = 1.0
    control.orientation.y = 0.0
    control.orientation.z = 0.0
    normalizeQuaternion(control.orientation)
    control.name = 'move_x'
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1.0
    control.orientation.x = 0.0
    control.orientation.y = 1.0
    control.orientation.z = 0.0
    normalizeQuaternion(control.orientation)
    control.name = 'move_z'
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1.0
    control.orientation.x = 0.0
    control.orientation.y = 0.0
    control.orientation.z = 1.0
    normalizeQuaternion(control.orientation)
    control.name = 'move_y'
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)

    # add the interactive marker to our collection &
    # tell the server to call processFeedback() when feedback arrives for it
    server.insert(int_marker, feedback_callback=processFeedback)

    # 'commit' changes and send to all clients
    server.applyChanges()

    rclpy.spin(node)
    server.shutdown()