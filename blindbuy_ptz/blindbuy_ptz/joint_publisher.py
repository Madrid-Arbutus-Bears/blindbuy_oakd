import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


class JointPublisher(Node):

    def __init__(self):
        super().__init__('joint_publisher')
        self.joint_pub = self.create_publisher("joint_states", JointState, 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        joint_msg=JointState()
        #joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name[0]='ptz_pan_joint'
        joint_msg.position[0]=0.5
        joint_msg.name[1]='ptz_tilt_joint'
        joint_msg.position[1]=1.0
        self.joint_pub.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)

    joint_publisher = JointPublisher()

    rclpy.spin(joint_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    joint_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()