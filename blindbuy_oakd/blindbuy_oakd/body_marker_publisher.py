import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from visualization_msgs.msg import Marker, MarkerArray


class BodyMarkerPublisher(Node):

    def __init__(self):
        super().__init__('body_marker_publisher')
        self.publisher_ = self.create_publisher(Marker, 'body_marker', 1)
        timer_period = 0.033  # 30Hz=30FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        marker=Marker()
        marker.id=0
        marker.header.frame_id='head'
        marker.type=Marker.MESH_RESOURCE
        marker.mesh_resource = "package://blindbuy_oakd/meshes/head.dae"
        marker.action=Marker.ADD
        marker.pose.position.x=0.0
        marker.pose.position.y=0.0
        marker.pose.position.z=0.0
        #https://wiki.ogre3d.org/tiki-index.php?page=Quaternion%20and%20Rotation%20Primer#Some_useful_normalized_quaternions
        marker.pose.orientation.x=0.0
        marker.pose.orientation.y=0.0
        marker.pose.orientation.z=0.0
        marker.pose.orientation.w=1.0
        marker.scale.x=0.07
        marker.scale.y=0.07
        marker.scale.z=0.07
        #Skin color
        # marker.color.r=0.98824
        # marker.color.g=0.81569
        # marker.color.b=0.70588
        marker.color.r=1.0
        marker.color.g=1.0
        marker.color.b=1.0
        marker.color.a=1.0
        self.publisher_.publish(marker)
   




def main(args=None):
    rclpy.init(args=args)

    body_marker_publisher = BodyMarkerPublisher()

    rclpy.spin(body_marker_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    body_marker_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()