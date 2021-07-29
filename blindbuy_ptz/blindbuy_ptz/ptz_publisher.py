import rclpy
from rclpy.node import Node
from math import pi

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped


class PTZPublisher(Node):

    def __init__(self):
        super().__init__('ptz_publisher')
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.subscription = self.create_subscription(PointStamped,'head_detection',self.listener_callback,1)
        self.pan_position=0.0
        self.tilt_position=0.0
        #Initialize position to the center
        self.publish_joints(-pi/2,0.0)


    def listener_callback(self, msg):
        x=self.limit_value(msg.point.x, min=-200, max=200)
        y=self.limit_value(msg.point.y, min=-150, max=150)
        pan=self.pan_position
        tilt=self.tilt_position

        if x>10:
            pan+=abs(x/10000) #Pan is proportional to the distance from the center
        elif x<-10:
            pan-=abs(x/10000)

        if y>25:
            tilt-=abs(y/10000)
        elif y<25:
            tilt+=abs(y/10000)

        self.publish_joints(pan,tilt)

    def limit_value(self,value, min, max):
        if value>max:
            value=max
        if value<min:
            value=min
        return value



    def publish_joints(self, pan, tilt):
        self.pan_position=self.limit_value(pan, -pi, 0.0)
        self.tilt_position=self.limit_value(tilt, -pi/4, pi/4)
        joint_msg=JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name=['ptz_pan','ptz_tilt']
        joint_msg.position=[self.pan_position, self.tilt_position]
        self.joint_pub.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)

    ptz_publisher = PTZPublisher()

    rclpy.spin(ptz_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ptz_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()