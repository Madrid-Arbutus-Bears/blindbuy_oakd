import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


class JointPublisher(Node):

    def __init__(self):
        super().__init__('joint_publisher')
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        timer_period = 0.01  # 10 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.direction=1
        self.angle=0.0

    def timer_callback(self):
        if self.angle>=3.14:
            self.direction=-1
        if self.angle<=-3.14:
            self.direction=1

        joint_msg=JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name=['ptz_pan_joint','ptz_tilt_joint']
        joint_msg.position=[self.angle, self.angle/2]
        self.joint_pub.publish(joint_msg)

        if self.direction==1:
            self.angle+=0.005
        else:
            self.angle-=0.005
        


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