# BlindBuy: Shopping Assistant for Visually Impaired People

## Installation (tested on Ubuntu 20.04 - ROS 2 Foxy)

[Install ROS2 Foxy](https://docs.ros.org/en/foxy/Installation/Linux-Install-Debians.html)

Don't forget to install colcon:
```
sudo apt install python3-colcon-common-extensions
```
Add following lines in ~/.bashrc:
```
source /opt/ros/foxy/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/ros2_ws
source ~/ros2_ws/install/setup.bash
```
Clone repository:
```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone -b python-api https://github.com/DaniGarciaLopez/blindbuy_oakd.git --recursive
```
[Official installation guide](https://docs.luxonis.com/projects/api/en/latest/install/):
```
sudo wget -qO- http://docs.luxonis.com/_static/install_dependencies.sh | bash
python3 -m pip install depthai
```
Install pip requirements:
```
cd ~/ros2_ws/src/blindbuy_oakd/blindbuy_oakd/depthai-python/examples
python3 install_requirements.py
```
Install ROS packages:
```
sudo apt install ros-foxy-joint-state-publisher
sudo apt install ros-foxy-joint-state-publisher-gui
```

Build:
```
cd ~/ros2_ws/
colcon build
```

## How to start
```
ros2 launch blindbuy_oakd demo.launch.py
```
## Hardware Reference Frame (TF)
### RPLidar A1
![image](https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd/blob/python-api/blindbuy_oakd/data/images/rplidar_a1.png)
### OpenAL
![image](https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd/blob/python-api/blindbuy_oakd/data/images/OpenAL.png)
