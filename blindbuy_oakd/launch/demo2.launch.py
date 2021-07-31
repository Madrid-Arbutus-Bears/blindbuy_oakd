#!/usr/bin/env python3
#
# Copyright 2020, Ben Bongalon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ben Bongalon (ben.bongalon@gmail.com)

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    pkg_blindbuy_description = get_package_share_directory('blindbuy_description')
    rviz_file_name = 'demo.rviz'

    rviz = os.path.join(get_package_share_directory('blindbuy_oakd'),'rviz',rviz_file_name)

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_blindbuy_description, 'launch', 'model_joints.launch.py')
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='blindbuy_oakd',
            executable='face_detection',
            name='face_detection',
            output='screen',
            parameters=[{}],
        ),
        Node(
            package='blindbuy_oakd',
            executable='body_marker_publisher',
            name='body_marker_publisher',
            output='screen',
            parameters=[{}],
        ),
        Node(
            package='blindbuy_servers',
            executable='product_distance_server',
            name='product_distance_server',
            output='screen',
            parameters=[{}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz]),
    ])
