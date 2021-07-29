import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    urdf_file_name = 'cart.urdf'

    print("urdf_file_name : {}".format(urdf_file_name))

    urdf_path = os.path.join(get_package_share_directory('blindbuy_description'),'urdf',urdf_file_name)
    urdf = open(urdf_path).read()

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': urdf,
            }],
        ),
        #https://answers.ros.org/question/357042/joint_state_publisher-waiting-for-robot_description-to-be-published-on-the-robot_description-topic/
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            arguments=[urdf_path],
            parameters=[{
                # 'rate': 30, #30Hz Default:10Hz
                'zeros': {'ptz_pan': -1.5708} #Modify default position
            }],
        ),
    ])