# blindbuy_oakd

## Installation
### Prerrequesites
```
sudo apt install libopencv-dev
sudo apt install python3-vcstool
```
```
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
```
```
sudo apt-get install -y ros-foxy-ros1-bridge
```
Recommended to install terminator:
```
sudo apt-get install -y terminator
```
Add in ~/.bashrc  
```
#ROS Noetic
source ~/catkin_ws/devel/setup.bash

#ROS 2 Foxy
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/ros2_ws
source ~/ros2_ws/install/setup.bash
```
### ROS Noetic
#### Clone repository
Clone noetic-devel branch:
```
cd ~/catkin_ws/src
git clone -b noetic-devel --recursive https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd.git
```
Add in ~/.bashrc     
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
#### Install depthai
The following script will install depthai-core and update usb rules and install depthai devices

```
sudo wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/noetic-devel/install_dependencies.sh | sudo bash
```
```
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
wget https://raw.githubusercontent.com/luxonis/depthai-ros/noetic-devel/underlay.repos
vcs import src < underlay.repos
rosdep install --from-paths src --ignore-src -r -y
catkin_make
```
#### JSK Visualization Plugin - (https://github.com/jsk-ros-pkg/jsk_visualization)
```
sudo apt-get install ros-noetic-jsk-visualization
```
### ROS 2 Foxy
#### Clone repository
Clone foxy-devel branch:
```
cd ~/ros2_ws/src
git clone -b foxy-devel --recursive https://github.com/Madrid-Arbutus-Bears/blindbuy_oakd.git
```
#### Install depthai
The following script will install depthai-core and update usb rules and install depthai devices
```
sudo wget -qO- https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/install_dependencies.sh | sudo bash
```
```
mkdir -p ~/ros2_ws/src
cd  ~/ros2_ws
wget https://raw.githubusercontent.com/luxonis/depthai-ros/foxy-devel/underlay.repos
vcs import src < underlay.repos
rosdep install -y -r -q --from-paths src --ignore-src --rosdistro foxy
colcon build
```
### Install drivers

[Install CH340 driver](https://learn.sparkfun.com/tutorials/how-to-install-ch340-drivers/drivers-if-you-need-them)

Python libraries:
```
pip3 install pyserial
```
Copy rules for usb devices:
```
sudo cp ~/ros2_ws/src/blindbuy_oakd/cfg/99-usb.rules /etc/udev/rules.d/
```
[Find usb devices name:](https://unix.stackexchange.com/questions/144029/command-to-determine-ports-of-a-device-like-dev-ttyusb0)
```
#!/bin/bash

for sysdevpath in $(find /sys/bus/usb/devices/usb*/ -name dev); do
    (
        syspath="${sysdevpath%/dev}"
        devname="$(udevadm info -q name -p $syspath)"
        [[ "$devname" == "bus/"* ]] && exit
        eval "$(udevadm info -q property --export -p $syspath)"
        [[ -z "$ID_SERIAL" ]] && exit
        echo "/dev/$devname - $ID_SERIAL"
    )
done
```
```
sudo chmod 666 /dev/ttyUSB0
```
## Launch
### ROS Bridge
Source:
```
source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash
```
Run:
```
ros2 run ros1_bridge dynamic_bridge
```
### ROS Noetic
Source:
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
1st terminal:
```
roscore
```
2nd terminal:
```
roslaunch depthai_examples mobile_publisher.launch
```
### ROS 2 Foxy
Source:
```
source /opt/ros/foxy/setup.bash
source /usr/share/colcon_cd/function/colcon_cd.sh
export _colcon_cd_root=~/ros2_ws
source ~/ros2_ws/install/setup.bash
```





